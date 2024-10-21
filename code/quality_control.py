from PIL import Image
import os

from datetime import datetime
from aind_data_schema_models.modalities import Modality
from aind_data_schema.core.quality_control import QualityControl, QCEvaluation, QCMetric, QCStatus, Status, Stage
from aind_qcportal_schema.metric_value import CheckboxMetric


def define_registration_summary_qcmetric_for_plane(plane_id):

    metric = QCMetric(
        name=f'{plane_id} Registration Summary',
        description=f'The registration summary output by suite2p for plane {plane_id}.',
        # Using path to png within motion correction capsule results
        reference='/results/{plane_id}/motion_correction/{plane_id}_registration_summary.png',
        value=CheckboxMetric(
            value="Placeholder CheckboxMetric Value",
            # Possible options for the metric
            options=[
                'Unresonable motion',
                'No motion',
                'Other Issue with Motion Correction'
                ],
            status=[
                Status.FAIL,
                Status.PASS,
                Status.PENDING #TODO, what should this be?, pending allowed here?
                ],
            status_history=[                                
                QCStatus(
                    evaluator='Initial Pending Status',
                    timestamp=datetime.now(), #TODO: Use same timestamp for all metrics?
                    # Requires manual annotation
                    status=Status.PENDING
                )
            ]
        ),
    )

def define_fov_quality_qcmetric_for_plane(plane_id):
        metric = QCMetric(
        name=f'{plane_id} Registration Summary',
        description=f'The registration summary output by suite2p for plane {plane_id}.',
        # Using path to png within motion correction capsule results
        reference='/results/{plane_id}/motion_correction/{plane_id}_registration_summary.png',
        value=CheckboxMetric(
            value="Placeholder CheckboxMetric Value",
            # Possible options for the metric
            options=[
                'Uncorrected motion present',
                'Low signal-to-noise ratio',
                'De-warping Vertical Banding Artifact',
                'Laser/scanner interference',
                'No cells in FOV',
                'Other Issue with FOV Quality'
                ],
            status=[
                Status.FAIL,
                Status.PASS,
                Status.PASS,
                Status.PASS,
                Status.PASS,
                Status.PENDING #TODO, what should this be?
                ],
            status_history=[                                
                QCStatus(
                    evaluator='Initial Pending Status',
                    timestamp=datetime.now(), #TODO: Use same timestamp for all metrics?
                    # Requires manual annotation
                    status=Status.PENDING
                )
            ]
        ),
    )
        
def combine_and_save_max_and_average_projection_for_plane(plane_id):
    '''
    Combine the max and average projections for a plane into a single image and save it to the motion correction capsule results directory.

    Args:
        plane_id (str or int): The plane ID for the motion correction data.
    
    Returns:
        str: The path to the combined image file.
    '''

    max_projection_png_path = f'/results/{plane_id}/motion_correction/{plane_id}_max_projection.png'
    average_projection_png_path = f'/results/{plane_id}/motion_correction/{plane_id}_average_projection.png'

    # Combine the max and average projections into a single image using PIL
    max_projection = Image.open(max_projection_png_path)
    average_projection = Image.open(average_projection_png_path)

    # Create a new image with the max projection on right and the average projection on left with a 10px white border around each
    combined_image = Image.new('RGB', (max_projection.width + average_projection.width + 40, max_projection.height + 20), (255, 255, 255))
    combined_image.paste(average_projection, (10, 10))
    combined_image.paste(max_projection, (average_projection.width + 30, 10))

    #TODO: Add "Max Projection" and "Average Projection" labels to the image

    # Save the combined image to the motion correction capsule results directory
    combined_image.save(f'/results/{plane_id}/motion_correction/{plane_id}_combined_projections.png')
    return f'/results/{plane_id}/motion_correction/{plane_id}_combined_projections.png'

def get_plane_ids():
    # Get the plane IDs from the motion correction data results directory
    # Any folder in /results/ is a plane ID

    plane_ids = []
    for folder in os.listdir('/results/'):
        if os.path.isdir(f'/results/{folder}'):
            plane_ids.append(folder)
    
    return plane_ids

def create_and_write_quality_control_json():
    # Create a QualityControl object 

    plane_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    qc_metrics = []

    registration_summary_metrics = []
    fov_quality_metrics = []

    for plane_id in plane_ids:
        # Define the QC metrics for the plane
        registration_summary_metrics.append(define_registration_summary_qcmetric_for_plane(plane_id))

        combine_and_save_max_and_average_projection_for_plane(plane_id)
        fov_quality_metrics.append(define_fov_quality_qcmetric_for_plane(plane_id))

    qc_metrics.append(registration_summary_metrics)
    qc_metrics.append(fov_quality_metrics)

    qc = QualityControl(
        notes='This object represents the quality control for the motion-correction processing step.',
        evaluations=[
            QCEvaluation(
                name="Registration Summary",
                description="Review the registration summary output by suite2p to ensure that the motion correction was successful.",
                stage=Stage.PROCESSED,
                modality=Modality.from_abbreviation('pophys'),
                notes="",
                allow_failed_metrics=False,
                metrics=registration_summary_metrics
            ),
            QCEvaluation(
                name="FOV Quality Summary",
                description="Review the average and max projections to ensure that the FOV quality is sufficient.",
                stage=Stage.PROCESSED,
                modality=Modality.from_abbreviation('pophys'),
                notes="",
                allow_failed_metrics=False,
                metrics=registration_summary_metrics
            )
        ]
    )

    qc.write_standard_file(output_directory=output_directory)
