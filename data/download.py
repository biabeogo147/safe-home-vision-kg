"""Enhanced Open Images downloader with ontology-based filtering."""

import argparse
from concurrent import futures
import os
import re
import sys
import json
import pandas as pd

import boto3
import botocore
import tqdm

BUCKET_NAME = 'open-images-dataset'
REGEX = r'(test|train|validation|challenge2018)/([a-fA-F0-9]*)'

class OntologyFilteredDownloader:
    """Downloads Open Images with ontology-based filtering."""

    def __init__(self, ontology_path='../reasoning/ontology.json'):
        """Initialize downloader with ontology.

        Args:
            ontology_path: Path to ontology.json file
        """
        with open(ontology_path, 'r', encoding='utf-8') as f:
            self.ontology = json.load(f)

        # Get the relevant class names from ontology
        self.relevant_classes = list(self.ontology['is_a'].keys())

    def filter_images_by_class(self, annotations_file, class_desc_file):
        """Filter image IDs to only include those with relevant classes.

        Args:
            annotations_file: Path to Open Images annotations CSV
            class_desc_file: Path to class-descriptions-boxable.csv

        Returns:
            Set of image IDs containing relevant classes
        """
        # Load class descriptions
        class_desc = pd.read_csv(class_desc_file, names=['class_id', 'class_name'])

        # Create mapping from class name to class_id
        name_to_id = dict(zip(class_desc['class_name'], class_desc['class_id']))

        # Get class_ids for relevant classes
        relevant_class_ids = []
        for class_name in self.relevant_classes:
            if class_name in name_to_id:
                relevant_class_ids.append(name_to_id[class_name])

        # Load annotations
        annotations = pd.read_csv(annotations_file)

        # Filter annotations
        filtered_annotations = annotations[
            annotations['LabelName'].isin(relevant_class_ids)
        ]

        # Get unique image IDs
        image_ids = set(filtered_annotations['ImageID'].unique())

        print(f"Found {len(image_ids)} images containing relevant classes")
        print(f"Relevant classes: {self.relevant_classes}")

        return image_ids


def check_and_homogenize_one_image(image):
    split, image_id = re.match(REGEX, image).groups()
    yield split, image_id


def check_and_homogenize_image_list(image_list):
    for line_number, image in enumerate(image_list):
        try:
            yield from check_and_homogenize_one_image(image)
        except (ValueError, AttributeError):
            raise ValueError(
                f'ERROR in line {line_number} of the image list. The following image '
                f'string is not recognized: "{image}".')


def read_image_list_file(image_list_file):
    with open(image_list_file, 'r') as f:
        for line in f:
            yield line.strip().replace('.jpg', '')


def download_one_image(bucket, split, image_id, download_folder):
    try:
        bucket.download_file(f'{split}/{image_id}.jpg',
                            os.path.join(download_folder, f'{image_id}.jpg'))
    except botocore.exceptions.ClientError as exception:
        sys.exit(
            f'ERROR when downloading image `{split}/{image_id}`: {str(exception)}')


def download_all_images(args):
    """Downloads all images specified in the input file."""
    bucket = boto3.resource(
        's3', config=botocore.config.Config(
            signature_version=botocore.UNSIGNED)).Bucket(BUCKET_NAME)

    download_folder = args['download_folder'] or os.getcwd()

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    try:
        image_list = list(
            check_and_homogenize_image_list(
                read_image_list_file(args['image_list'])))
    except ValueError as exception:
        sys.exit(exception)

    progress_bar = tqdm.tqdm(
        total=len(image_list), desc='Downloading images', leave=True)
    with futures.ThreadPoolExecutor(
        max_workers=args['num_processes']) as executor:
        all_futures = [
            executor.submit(download_one_image, bucket, split, image_id,
                            download_folder) for (split, image_id) in image_list
        ]
        for future in futures.as_completed(all_futures):
            future.result()
            progress_bar.update(1)
    progress_bar.close()


def download_filtered_images(args):
    """Downloads images filtered by ontology classes."""
    downloader = OntologyFilteredDownloader(args['ontology'])

    # Filter images based on ontology
    image_ids = downloader.filter_images_by_class(
        args['annotations'], args['class_descriptions'])

    # Create image list in the format split/image_id
    image_list = []
    for split in ['train', 'validation', 'test']:
        # You would need to determine which split each image belongs to
        # For simplicity, we'll use train split
        for image_id in image_ids:
            image_list.append(f'{split}/{image_id}')

    args['image_list'] = image_list
    download_all_images(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'image_list',
        type=str,
        nargs='?',
        default=None,
        help='Filename that contains the split + image IDs of the images to download.')
    parser.add_argument(
        '--num_processes',
        type=int,
        default=5,
        help='Number of parallel processes to use (default is 5).')
    parser.add_argument(
        '--download_folder',
        type=str,
        default=None,
        help='Folder where to download the images.')
    parser.add_argument(
        '--annotations',
        type=str,
        default=None,
        help='Path to annotations CSV for filtering')
    parser.add_argument(
        '--class_descriptions',
        type=str,
        default=None,
        help='Path to class-descriptions-boxable.csv')
    parser.add_argument(
        '--ontology',
        type=str,
        default='../reasoning/ontology.json',
        help='Path to ontology.json')
    parser.add_argument(
        '--filter_by_ontology',
        action='store_true',
        help='Filter images by ontology classes')

    args = parser.parse_args()

    if args.filter_by_ontology:
        if not args.annotations or not args.class_descriptions:
            print("Error: --annotations and --class_descriptions are required when using --filter_by_ontology")
            sys.exit(1)
        download_filtered_images(vars(args))
    else:
        download_all_images(vars(args))