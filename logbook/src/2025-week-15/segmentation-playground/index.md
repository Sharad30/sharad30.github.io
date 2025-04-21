# segmentation-playground

https://github.com/soumendra/segmentation-playground

* Built an image segmentation pipeline (using `pytorch`) to train Mask RCNN model.
* Pascal VOC dataset was used and instead of taking all the classes to fine tune, the pipeline was setup to classify beween person and background (all the other classes were labelled background)
* The pipeline was tested on local machine (with limited GPU). Even though the pipeline was running, the model was not learning at all with a very high training loss.
* To get the model up and running, the decision to migrate the project to `ultralytics` was taken.
    - Need to get back to test the `pytorch` pipeline again and figure out the issues.