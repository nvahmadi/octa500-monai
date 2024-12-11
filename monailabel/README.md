# Folder setup for MONAI Label OCTA-500 Reproducer

## Download the OCTA-500 Nifti data
Benjamin has shared access to a Google Drive folder, where he can download the data and provide it to you internally. The data is modified (Nifit-fied) from a CC-BY 4.0 licensed dataset, so it is safe to use. Please make sure to cite the authors appropriately, to attribute their contribution (see `README.md` in the dataset folder).

Once downloaded please unzip it to a working directory (assumed as `/workdir/Projects` from now on), into a subfolder `datasets`, such that you obtain the following folder structure:

```
/workdir/Projects/datasets/OCTA-500/
├── OCTA-500 arXiv (2012.07261v3).pdf
├── OCTA500_MONAI_3mm
├── OCTA500_MONAILABEL_3mm
├── Projection Maps
├── README.md
├── Text labels.csv
└── Text labels.xlsx
```

## Pull the GitHub repo
In your folder `/workdir/Projects`, run:

```
git clone https://github.com/nvahmadi/octa500-monai.git
```

This should give you the following folder structure:
```
/workdir/Projects/octa500-monai/
├── additional_tutorials
├── copy_to_monailabel_dataset.ipynb
├── LICENSE
├── monailabel
├── octa500_segmentation_3d.ipynb
├── pyocta500
├── README.md
└── slicer_load_oct500_subject.py
```

## Docker run monailabel container
Run `start_docker_monailabel_latest.sh`.

In line 23 of the file `start_docker_monailabel_latest.sh`, you can see that we map the data resource:

`-v /workdir:/data/`

Please don't forget to adapt `/workdir` to your project working directory. Inside the folder you already have unzipped data into the `/workdir/Projects/datasets` folder, and you should find the code in `/workdir/Projects/octa500-monai`, pulled from Github.

## Copy the OCTA-500 files for MONAI Label
MONAI Label assumes a certain folder structure for files. Inside the container, start a jupyter lab, and run the notebook:
```
/data/Projects/octa500-monai/copy_to_monailabel_dataset.ipynb
```
This will copy files from:<br>
`/data/Projects/datasets/OCTA-500/`<br>
into:<br>
`/data/Projects/datasets/OCTA-500/OCTA500_MONAILABEL_3mm`<br>
and rename images, groundtruth annotations, and place them into the right sub-folders.

## Start MONAI Label server inside the container
Inside the container, run this script to startup MONAI Label server:
```
/data/Projects/octa500-monai/monailabel/start_monailabel_server.sh
```
This will execute this command, to run the pre-trained model:
```
monailabel start_server \
    --app /data/Projects/octa500-monai/monailabel/apps/octa500_100train_800epochs \
    --studies /data/Projects/datasets/OCTA-500/OCTA500_MONAILABEL_3mm \
    --conf models segmentation \
    --conf use_pretrained_model false
```
You can find this folder `octa500_100train_800epochs` inside the downloaded dataset folder. Please place it into the designated location.

