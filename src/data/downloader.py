"""
MovieLens Dataset Downloader.

Handles automatic downloading, extraction, and validation of the MovieLens
dataset. Designed to be idempotent — re-running it skips download if files
already exist and pass validation.

Pipeline:
    1. Check if raw data files already exist (skip if so)
    2. Download the zip archive with a tqdm progress bar
    3. Verify the downloaded file is a valid ZIP
    4. Extract to data/raw/ directory
    5. Validate that all expected CSV files are present

Usage:
    from src.data.downloader import MovieLensDownloader
    downloader = MovieLensDownloader()
    downloader.download()
"""

import zipfile
from pathlib import Path
from typing import List, Optional

import requests
from tqdm import tqdm

from src.utils.exceptions import DataDownloadError, DataNotFoundError
from src.utils.helpers import ensure_dir, load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Expected files inside the extracted MovieLens directory
EXPECTED_FILES: List[str] = ["ratings.csv", "movies.csv", "tags.csv", "links.csv"]


class MovieLensDownloader:
    """
    Downloads and extracts the MovieLens dataset.

    Attributes:
        cfg: OmegaConf configuration object.
        raw_dir: Path to data/raw/ directory.
        dataset_name: Name of the MovieLens dataset (e.g. "ml-latest-small").
        dataset_dir: Path to the extracted dataset directory.
        download_url: Remote URL for the dataset ZIP file.

    Example:
        >>> downloader = MovieLensDownloader()
        >>> downloader.download()
        >>> print(downloader.get_data_paths())
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialise the downloader with configuration.

        Args:
            config_path: Path to config.yaml. Uses default if None.
        """
        self.cfg = load_config(config_path)
        self.raw_dir: Path = resolve_path(self.cfg.paths.raw_dir)
        self.dataset_name: str = self.cfg.data.dataset_name
        self.dataset_dir: Path = self.raw_dir / self.dataset_name
        self.download_url: str = self.cfg.data.download_url
        self._zip_path: Path = self.raw_dir / f"{self.dataset_name}.zip"

    def is_downloaded(self) -> bool:
        """
        Check whether the dataset has already been downloaded and extracted.

        Returns:
            True if all expected CSV files exist in the dataset directory.
        """
        if not self.dataset_dir.exists():
            return False
        for filename in EXPECTED_FILES:
            if not (self.dataset_dir / filename).exists():
                logger.warning(f"Missing expected file: {filename}")
                return False
        return True

    def download(self, force: bool = False) -> Path:
        """
        Download and extract the MovieLens dataset.

        Idempotent: if all files are already present and ``force=False``,
        this method logs a message and returns immediately without any
        network requests.

        Args:
            force: If True, re-download even if files already exist.

        Returns:
            Path to the extracted dataset directory.

        Raises:
            DataDownloadError: If the download or extraction fails.
        """
        if not force and self.is_downloaded():
            logger.info(
                f"Dataset already downloaded at: {self.dataset_dir}. "
                "Skipping download. Use force=True to re-download."
            )
            return self.dataset_dir

        ensure_dir(self.raw_dir)
        logger.info(f"Downloading MovieLens dataset from: {self.download_url}")

        self._download_zip()
        self._extract_zip()
        self._validate_files()

        logger.info(f"Dataset ready at: {self.dataset_dir}")
        return self.dataset_dir

    def _download_zip(self) -> None:
        """
        Download the ZIP archive with a streaming progress bar.

        Raises:
            DataDownloadError: On network error or non-200 HTTP response.
        """
        try:
            response = requests.get(self.download_url, stream=True, timeout=60)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise DataDownloadError(
                f"Failed to reach download URL: {self.download_url}",
                details=str(exc),
            ) from exc

        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 8192  # 8 KB chunks

        logger.info(
            f"Total size: {total_size / 1_048_576:.1f} MB — saving to {self._zip_path}"
        )

        try:
            with open(self._zip_path, "wb") as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                colour="green",
            ) as progress:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        progress.update(len(chunk))
        except OSError as exc:
            raise DataDownloadError(
                f"Failed to write ZIP file to {self._zip_path}",
                details=str(exc),
            ) from exc

        logger.info(f"Download complete: {self._zip_path}")

    def _extract_zip(self) -> None:
        """
        Extract the downloaded ZIP archive to the raw data directory.

        Raises:
            DataDownloadError: If the file is not a valid ZIP archive.
        """
        if not zipfile.is_zipfile(self._zip_path):
            raise DataDownloadError(
                f"Downloaded file is not a valid ZIP archive: {self._zip_path}",
                details="The file may be corrupted or the download was interrupted.",
            )

        logger.info(f"Extracting {self._zip_path} → {self.raw_dir}")
        try:
            with zipfile.ZipFile(self._zip_path, "r") as zip_ref:
                members = zip_ref.namelist()
                logger.debug(f"ZIP contains {len(members)} files")
                for member in tqdm(members, desc="Extracting", colour="blue"):
                    zip_ref.extract(member, self.raw_dir)
        except zipfile.BadZipFile as exc:
            raise DataDownloadError(
                "Extraction failed — ZIP archive appears corrupted.",
                details=str(exc),
            ) from exc

        # Clean up the zip file after successful extraction
        self._zip_path.unlink(missing_ok=True)
        logger.info("ZIP archive removed after extraction.")

    def _validate_files(self) -> None:
        """
        Verify all expected CSV files exist in the extracted directory.

        Raises:
            DataNotFoundError: If any expected file is missing.
        """
        missing = []
        for filename in EXPECTED_FILES:
            filepath = self.dataset_dir / filename
            if not filepath.exists():
                missing.append(filename)

        if missing:
            raise DataNotFoundError(
                f"Extraction complete but {len(missing)} expected files are missing.",
                details=f"Missing: {missing}. Expected in: {self.dataset_dir}",
            )

        logger.info(
            f"Validation passed — all {len(EXPECTED_FILES)} files present: "
            + ", ".join(EXPECTED_FILES)
        )

    def get_data_paths(self) -> dict:
        """
        Return a dictionary of file paths for all dataset CSV files.

        Returns:
            Dictionary mapping file keys to absolute Path objects.

        Raises:
            DataNotFoundError: If dataset has not been downloaded yet.
        """
        if not self.is_downloaded():
            raise DataNotFoundError(
                "Dataset not found. Run download() first.",
                details=f"Expected at: {self.dataset_dir}",
            )

        return {
            "ratings": self.dataset_dir / self.cfg.data.ratings_file,
            "movies": self.dataset_dir / self.cfg.data.movies_file,
            "tags": self.dataset_dir / self.cfg.data.tags_file,
            "links": self.dataset_dir / self.cfg.data.links_file,
        }

    def print_dataset_info(self) -> None:
        """Log a summary of the downloaded dataset files and their sizes."""
        if not self.is_downloaded():
            logger.warning("Dataset not yet downloaded.")
            return

        paths = self.get_data_paths()
        logger.info("── Dataset File Summary ──────────────────────────")
        for key, path in paths.items():
            size_kb = path.stat().st_size / 1024
            logger.info(f"  {key:10s}: {path.name} ({size_kb:.1f} KB)")
        logger.info("─────────────────────────────────────────────────")


if __name__ == "__main__":
    downloader = MovieLensDownloader()
    downloader.download()
    downloader.print_dataset_info()
