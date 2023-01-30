package com.example.aps1

import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.android.gms.vision.face.FaceDetector
import com.otaliastudios.cameraview.controls.Mode
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.ByteArrayOutputStream
import java.lang.Float.min
import kotlin.math.ceil



class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val spinner = findViewById<ProgressBar>(R.id.progressBar)

        cameraView.setLifecycleOwner(this)

        val button = findViewById<Button>(R.id.get_breed)
        button.setOnClickListener {
            spinner.setVisibility(View.VISIBLE);
            cameraView.addFrameProcessor{ frame ->
                cameraView.close()
                val matrix = Matrix()
                matrix.setRotate(frame.rotationToUser.toFloat())

                if (frame.dataClass === ByteArray::class.java){
                    val out = ByteArrayOutputStream()
                    val yuvImage = YuvImage(
                        frame.getData(),
                        ImageFormat.NV21,
                        frame.size.width,
                        frame.size.height,
                        null
                    )
                    yuvImage.compressToJpeg(
                        Rect(0, 0, frame.size.width, frame.size.height), 100, out
                    )
                    val imageBytes = out.toByteArray()
                    var bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

                    bitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                    bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                    processBitmap(bitmap)
                } else {
                    Toast.makeText(this, "Camera Data not Supported", Toast.LENGTH_LONG).show()
                }
                spinner.setVisibility(View.INVISIBLE);
            }
        }
    }


    private fun processBitmap(bitmap: Bitmap){
        val label = predict(bitmap)
        val maxEntry = label.maxWith(Comparator { x, y -> x.value.compareTo(y.value)})
        //Log.d("Labels", label.toString())
        val prediction = findViewById<TextView>(R.id.label)
        val confidence = findViewById<TextView>(R.id.confidence)
        if(maxEntry?.value!! <0.2){
            prediction.text = "No dog found"
            confidence.text = "--"
        }else{
            prediction.text = maxEntry?.key.toString()
            confidence.text = (maxEntry?.value?.times(100)).toString()+" %"
        }
        Log.d("Data:", maxEntry.toString())
    }

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

}
