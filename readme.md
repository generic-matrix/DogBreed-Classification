# Dependencies

* Python 3.6
* TensorFlow 2.1
* Android Studio

# Outputs


German Shepard       |  Pug
:-------------------------:|:-------------------------:
![german_shepard](https://github.com/generic-matrix/DogBreed-Classification/blob/main/Output/german_shepard.png?raw=true)  |  ![Pug](https://github.com/generic-matrix/DogBreed-Classification/blob/main/Output/pug.png?raw=true)



# Project Structure

The project has 4 folders

* Android App -> The Android App code
* Labels -> It has labels.txt
* APK -> The apk.app which can be installed onto a compatible Android device
* Training -> It has Training.ipynb whci can be opened in Google Colab
* Output -> The Output Video is in here

# How to build the Android App

* git clone <>
* cd Android App (ALos download the model.tflite from the below link and paste it in the assets folder)
* Open the same on the Android Studio

# APK can be downloaded from the [Link](https://drive.google.com/file/d/1RM5MZsdvZRtQnZOu7a-nhcYXtnitGVya/view?usp=sharing)


# Steps Followed

We are using NASNetLarge model , refer training.ipynb in the training folder

1) convert classes.json to labels.txt and classes.json
```
where {"A": 0, "B": 1,"C":2}
```
we need to convert the above json  to labels.txt like below
```
A
B
C
```

```
import json
data=json.loads('<content in classes.json>')
text_file = open("labels.txt", "w")
#write string to file
text_file.write('\n'.join(data.keys()))
#close file
text_file.close()
```

2) Download model.tflite from [Link](https://drive.google.com/file/d/1mYw36XgrNCya98B2Vx1QU601_Q5YJJs4/view?usp=sharing) 


3) Create android app from android studio 

3) Add com.otaliastudios:cameraview and tflite as dependencies

```
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly'
    implementation 'org.tensorflow:tensorflow-lite-gpu:0.0.0-nightly'
    implementation 'org.tensorflow:tensorflow-lite-support:0.0.0-nightly'
    implementation 'com.github.esafirm.android-image-picker:imagepicker:2.3.1'
    implementation 'com.github.bumptech.glide:glide:4.5.0'
    implementation 'com.google.android.gms:play-services-vision:20.0.0'
    api 'com.otaliastudios:cameraview:2.6.2'
```


4) In the layout add

```
<com.camerakit.CameraKitView
    android:id="@+id/camera"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    android:adjustViewBounds="true"
    android:keepScreenOn="true"   <!-- keep screen awake while CameraKitView is active -->
    app:camera_flash="auto"
    app:camera_facing="back"
    app:camera_focus="continuous"
    app:camera_permissions="camera" />
```


5) Check activity_main.xml 

    5.1) Add CameraView , ProgressBar , A button and two text views

6) Paste labels.txt and model.tflite onto the assets folder 

7) Review MainActivity.kt

    7.1) Set the cameraview life cycle owner

    ```
    cameraView.setLifecycleOwner(this)
    ```

    7.2) Button onClick , add frameprocessor on the camera view

    ```
    val button = findViewById<Button>(R.id.get_breed)
        button.setOnClickListener {
            cameraView.addFrameProcessor{ frame ->
            }
    }
    ```

    7.3) From frame convert to bitmap and send it to predict function 

    where predit function is 

    ```
    private fun predict(input: Bitmap): MutableMap<String, Float> {
            // load model
            val modelFile = FileUtil.loadMappedFile(this, "model.tflite")
            val model = Interpreter(modelFile, Interpreter.Options()) 
            val labels = FileUtil.loadLabels(this, "labels.txt")

            // data type
            val imageDataType = model.getInputTensor(0).dataType() 
            val inputShape = model.getInputTensor(0).shape() 

            val outputDataType = model.getOutputTensor(0).dataType() 
            val outputShape = model.getOutputTensor(0).shape() 

            var inputImageBuffer = TensorImage(imageDataType)
            val outputBuffer = TensorBuffer.createFixedSize(outputShape, outputDataType) 

            // preprocess
            val cropSize = kotlin.math.min(input.width, input.height)
            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeWithCropOrPadOp(cropSize, cropSize)) 
                .add(ResizeOp(inputShape[1], inputShape[2], ResizeOp.ResizeMethod.NEAREST_NEIGHBOR)) 
                .add(NormalizeOp(127.5f, 127.5f)) 
                .build()

            // load image
            inputImageBuffer.load(input) 
            inputImageBuffer = imageProcessor.process(inputImageBuffer) 

            // run model
            model.run(inputImageBuffer.buffer, outputBuffer.buffer.rewind())

            // get output
            val labelOutput = TensorLabel(labels, outputBuffer) 

            val label = labelOutput.mapWithFloatValue
            return label
    }
    ```
    The predict function returs the probablity for each and every classes

    7.4) Parse the probablity key value pair

    ```
        val label = predict(bitmap)
        val maxEntry = label.maxWith(Comparator { x, y -> x.value.compareTo(y.value)})
        val prediction = findViewById<TextView>(R.id.label)
        val confidence = findViewById<TextView>(R.id.confidence)
        if(maxEntry?.value!! <0.2){
            prediction.text = "No dog found"
            confidence.text = "--"
        }else{
            prediction.text = maxEntry?.key.toString()
            confidence.text = (maxEntry?.value?.times(100)).toString()+" %"
        }
    ```
