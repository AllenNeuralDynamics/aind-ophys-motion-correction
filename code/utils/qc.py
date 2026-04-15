"""QC visual generation and metric serialization."""

import json
import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from utils.metadata_utils import (
    build_fov_quality_metric,
    build_registration_summary_metric,
)


def combine_images_with_individual_titles(
    image_path_1: Path,
    image_path_2: Path,
    output_path: Path,
    title1: str = "Image 1",
    title2: str = "Image 2",
) -> None:
    """Combine two images side by side with individual titles.

    Parameters
    ----------
    image_path_1 : Path
        Path to the first image.
    image_path_2 : Path
        Path to the second image.
    output_path : Path
        Path to save the combined image.
    title1 : str
        Title for the first image.
    title2 : str
        Title for the second image.
    """
    img1 = Image.open(image_path_1)
    img2 = Image.open(image_path_2)

    title_height = 40
    combined_width = img1.width + img2.width
    combined_height = max(img1.height, img2.height) + title_height

    combined_image = Image.new("RGB", (combined_width, combined_height), "white")
    draw = ImageDraw.Draw(combined_image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    title1_bbox = draw.textbbox((0, 0), title1, font=font)
    title1_width = title1_bbox[2] - title1_bbox[0]
    draw.text(
        ((img1.width - title1_width) // 2, 10), title1, fill="black", font=font
    )

    title2_bbox = draw.textbbox((0, 0), title2, font=font)
    title2_width = title2_bbox[2] - title2_bbox[0]
    draw.text(
        (img1.width + (img2.width - title2_width) // 2, 10),
        title2,
        fill="black",
        font=font,
    )

    combined_image.paste(img1, (0, title_height))
    combined_image.paste(img2, (img1.width, title_height))

    combined_image.save(output_path)


def write_registration_summary_metric(output_dir: Path) -> None:
    """Serialize the registration summary QCMetric.

    QCMetric is named 'registration_summary_metric.json' and is
    saved to the same directory as *_registration_summary.png.
    Ex: '/results/<unique_id>/motion_correction/'

    Parameters
    ----------
    output_dir : Path
        Top-level results directory to search for the summary PNG.
    """
    file_path = next(output_dir.rglob("*_registration_summary.png"))

    # Remove '/results' from file_path
    reference_filepath = Path(*file_path.parts[2:])
    unique_id = reference_filepath.parts[0]

    metric = build_registration_summary_metric(unique_id, str(reference_filepath))

    with open(
        Path(file_path.parent) / f"{unique_id}_registration_summary_metric.json", "w"
    ) as f:
        json.dump(json.loads(metric.model_dump_json()), f, indent=4)
    logging.info("Saved QC metric -> %s", file_path.parent / f"{unique_id}_registration_summary_metric.json")


def write_fov_quality_metric(output_dir: Path) -> None:
    """Serialize the FOV Quality QCMetric.

    QCMetric is named 'fov_quality_metric.json' and is
    saved to the same directory as *_maximum_projection.png.
    Ex: '/results/<unique_id>/motion_correction/'

    Parameters
    ----------
    output_dir : Path
        Top-level results directory to search for projection PNGs.
    """
    avg_projection_file_path = next(output_dir.rglob("*_average_projection.png"))
    max_projection_file_path = next(output_dir.rglob("*_maximum_projection.png"))

    file_path = Path(str(max_projection_file_path).replace("maximum", "combined"))

    combine_images_with_individual_titles(
        avg_projection_file_path,
        max_projection_file_path,
        file_path,
        title1="Average Projection",
        title2="Maximum Projection",
    )

    # Remove /results from file_path
    reference_filepath = Path(*file_path.parts[2:])
    unique_id = reference_filepath.parts[0]

    metric = build_fov_quality_metric(unique_id, str(reference_filepath))

    with open(
        Path(file_path.parent) / f"{unique_id}_fov_quality_metric.json", "w"
    ) as f:
        json.dump(json.loads(metric.model_dump_json()), f, indent=4)
    logging.info("Saved QC metric -> %s", file_path.parent / f"{unique_id}_fov_quality_metric.json")
