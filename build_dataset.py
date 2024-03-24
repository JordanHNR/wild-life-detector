"""
Define constants and functions to build an object detection dataset from
several camera traps datasets (LILA Camera Traps).

Constants:
	* DATASETS_COLUMNS (List[str]): List of dataset's features name.
	* LILA_REPOSITORY (str): LILA Camera Traps dataset repository name.
	* TRUST_REMOTE_CODE (bool): Whether or not to allow LILA dataset 
	using scripts.

Functions:
	* load_lila_datasets: Load a list of LILA camera traps datasets from 
	HuggingFace.
	* restructure_datasets: Apply changes on datasets columns to unify
	datasets structures.
	* concatenate_datasets: Concatenate the input list of datasets into 
	a single dataframe.
	* filter_species: Filter out unspecified species from input dataset.
	* split_dataset: Assign a set (train, validation or test) to each 
	data, given specified ratios for each set.

Data are from HuggingFace downloaded from HuggingFace.
https://huggingface.co/datasets/society-ethics/lila_camera_traps
"""

import datasets
import numpy as np
import requests

from io import BytesIO
from PIL import Image, ImageDraw
from typing import List


DATASET_COLUMNS = ["image", "taxonomy", "bboxes", 
					"seq_num_frames", "seq_id",
					"width", "height"]
LILA_REPOSITORY = "society-ethics/lila_camera_traps"
TRUST_REMOTE_CODE = True


def load_lila_datasets(names: List[str]) -> List[datasets.arrow_dataset.Dataset]:
	"""
	Load a list of LILA camera traps datasets from HuggingFace.
	An empty list will be returned if an error occurs during loading. 

	Args:
		* names: List of dataset names.
	Return:
		* datasets_list: List of loaded datasets.
	"""
    
	try:
		datasets_list = [datasets.load_dataset(LILA_REPOSITORY, 
												name, 
												split="train",
												trust_remote_code=True)
						for name in names]
		return datasets_list
	except:
		return []


def restructure_datasets(datasets_list: List[datasets.arrow_dataset.Dataset],
						 datasets_names: List[str]=None) -> List[datasets.arrow_dataset.Dataset]:
	"""
	Apply changes on datasets columns such that all input datasets have 
	the same structure. Because all LILA datasets have not the same
	features.
	Applied changes include:
		* Bounding box column rename
		* Taxonomy column rename
		* Add seq_id and seq_num_frames columns
	
    Dataset names are used to generate sequence ids for datasets that 
    have none. If no dataset names are provided, the missing sequence_id 
	will be set to None.


	Args:
		* datasets_list: Input list of datasets to restructure.
		* datasets_names: Input dataset's names (default: None).
	Return:
		* datasets_list: List of restructured datasets.
	"""

	for i in range(len(datasets_list)):
		d = datasets_list[i].flatten()

		# Rename columns
		if "bboxes.bbox" in d.features:
			d = d.rename_column("bboxes.bbox", "bboxes")
		elif "annotations.bbox" in d.features:
			d = d.rename_column("annotations.bbox", "bboxes")
		d = d.rename_column("annotations.taxonomy", "taxonomy")

		# Add sequence related column if needed
		if "seq_id" not in d.features:
			if not datasets_names:
				seq_id = [None for _ in range(len(d))]
			else:
				seq_id = [f"{datasets_names[i]}_{_}"
			              for _ in range(len(d))]
			seq_num = np.array([1 for _ in range(len(d))], dtype=np.int32)
			d = d.add_column("seq_id", seq_id)
			d = d.add_column("seq_num_frames",  seq_num)
		
		datasets_list[i] = d
	
	return datasets_list


def concatenate_datasets(datasets_list: List[datasets.arrow_dataset.Dataset]) -> datasets.arrow_dataset.Dataset:
	"""
	Concatenate the input list of datasets into a single dataframe.
	All datasets are expected to have the same structure. Make sure to 
	call restructure_dataset before concatenation.

	Args:
		* datasets_list: List of datasets.
	
	Return:
		* dataset: Concatenated datasets.
	"""

	dataset = [d.remove_columns([col for col in d.column_names 
							     if col not in DATASET_COLUMNS])
			   for d in datasets_list]
	dataset = datasets.concatenate_datasets(dataset)

	return dataset


def filter_species(dataset: datasets.arrow_dataset.Dataset,
				   species: List[str]) -> datasets.arrow_dataset.Dataset:
	"""
	Filter out unspecified species from input dataset.

	Args:
		* dataset: Input dataset.
		* species: Species to keep.

	Return:
		* dataset: Filtered dataset.
	"""
	
	taxonomy = dataset.features["taxonomy"][0]
	species_labels = [taxonomy["species"].str2int(_) 
				      for _ in taxonomy["species"].names
					  for s in species if s == _]
	dataset = dataset.filter(lambda x: x["taxonomy"][0]["species"] 
						     in species_labels)

	return dataset


def split_dataset(dataset: datasets.arrow_dataset.Dataset,
				  ratios: List[float]) -> datasets.arrow_dataset.Dataset:
	"""
	Assign a set (train, validation or test) to each data, given 
	specified ratios for each set.
	
	Currently, split is done by sequences. But, a future version of this
	function should also make sure that distributions of species are
	similar between sets.

	Args:
		* dataset: Input dataset to split.
		* ratios: List of ratio for each subset, expressed as a number
		          in [0, 1].
	
	Return:
		* dataset: Split dataset.
	"""

	# Shuffle sequences
	sequences = np.unique(dataset["seq_id"])
	np.random.shuffle(sequences)
	
	# Split sequences
	train_idx = int(ratios[0] * len(sequences))
	val_idx = train_idx + int(ratios[1] * len(sequences))
	train_seq = sequences[:train_idx]
	val_seq = sequences[train_idx: val_idx]

	# TODO: Make sure species distributions are similar between sets. 

	sets = ["train" if dataset[i]["seq_id"] in train_seq
		 	else "val" if dataset[i]["seq_id"] in val_seq
			else "test" for i in range(len(dataset))]
	dataset = dataset.add_column("set", sets)

	return dataset


def save_dataset():
	"""
	"""

	return 0


def show(item):
	try:
		response = requests.get(item["image"])
		img = Image.open(BytesIO(response.content))
		
		if item["annotations.bbox"]:
			draw = ImageDraw.Draw(img)
			
			bbox = item["annotations.bbox"][0]
			bbox = [(bbox[0], bbox[1]),
		   			(bbox[0] + bbox[2], bbox[1] + bbox[3])]
			draw.rectangle(bbox, outline="red", width=5)
		else:
			print("No bbox available")
		return img
	except:
		print("Error while loading image.")


if __name__ == "__main__":
	# TODO: Control these variables through command line arguments
	names = ["Missouri Camera Traps", 
		     "ENA24",
			 "NACTI",
			 "SWG Camera Traps",
			 "WCS Camera Traps",
			 "Caltech Camera Traps",
			 "Idaho Camera Traps"]
	species = ["ursus americanus", # American Black Bear
			   "ursus thibetanus", # Asian Black Bear
			   "ursidae", # Bear
			   "sus scrofa", # Wild Boar
			   "vulpes vulpes", # Red Fox
			   "urocyon cinereoargenteus", # Gray Fox
			   "macaca mulatta" # Rhesus Macaque
			  ]
	ratios = [0.7, 0.2, 0.1]

	datasets_list = load_lila_datasets(names)
		
	if datasets_list:
		datasets_list = restructure_datasets(datasets_list, names)
		dataset = concatenate_datasets(datasets_list)
		print(f"Dataset length before filtering: {len(dataset)}")
		dataset = filter_species(dataset, species)
		print(f"Dataset length after filtering: {len(dataset)}")
		dataset = split_dataset(dataset, ratios)

    # random_data = dataset[0]
    # print(random_data)
    # show(random_data).show()
