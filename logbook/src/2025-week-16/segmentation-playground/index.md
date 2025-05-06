# segmentation-playground

https://github.com/soumendra/segmentation-playground

* Built an image segmentation pipeline (using `ultralytics`) to train yolov8 instance segmentation model.
* Discovered something about `PASCAL VOC dataset`
    - The original mask format is in color format, due to which a raw conversion to yolo format was not working. Explain it detail, in a small blog post.
* After fixing the data format, ultralytics `tuner` was setup to be able to train experiemnts one after the another.
    - Still need to run experiemtns for various combination of learning rate and batch_size
* `wandb` was setup to log each experiments
