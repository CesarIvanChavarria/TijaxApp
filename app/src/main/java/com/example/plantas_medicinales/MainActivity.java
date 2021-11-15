package com.example.plantas_medicinales;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.drawable.AdaptiveIconDrawable;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.service.quickaccesswallet.GetWalletCardsRequest;
import android.text.method.ScrollingMovementMethod;
import android.util.DisplayMetrics;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.github.mikephil.charting.charts.BarChart;
import com.github.mikephil.charting.charts.HorizontalBarChart;
import com.github.mikephil.charting.components.XAxis;
import com.github.mikephil.charting.data.BarData;
import com.github.mikephil.charting.data.BarDataSet;
import com.github.mikephil.charting.data.BarEntry;
import com.github.mikephil.charting.formatter.IndexAxisValueFormatter;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    protected Interpreter tflite;
    private MappedByteBuffer tfliteModel;
    private TensorImage inputImageBuffer;
    private int imageSizeX;
    private int imageSizeY;
    private TensorBuffer outputProbabilityBuffer;
    private TensorProcessor probabilityProcessor;
    private static final float IMAGE_MEAN = 0.0f;
    private static final float IMAGE_STD= 1.0f;
    private static final float PROBABILITY_MEAN = 0.0f;
    private static final float PROBABILITY_STD = 255.0f;
    private Bitmap bitmap;
    private List<String> labels;
    private HorizontalBarChart mBarChat;
    ImageView imageView, ejemplo, imgwarning;
    Uri imageUri;
    Button btnClassify;
    TextView prediction;
    TextView titulo, cuerpo, infowarning;
    androidx.cardview.widget.CardView cardView, cardWarning;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = (ImageView) findViewById(R.id.imageView);
        btnClassify = (Button) findViewById(R.id.classify);
        prediction = (TextView) findViewById(R.id.predictionTxt);
        titulo = (TextView) findViewById(R.id.titulo);
        cuerpo = (TextView) findViewById(R.id.info);
        ejemplo = (ImageView) findViewById(R.id.muestra);
        cardView = (androidx.cardview.widget.CardView) findViewById(R.id.card);
        infowarning = (TextView) findViewById(R.id.infowarning);
        cardWarning = (androidx.cardview.widget.CardView) findViewById(R.id.warning);
        imgwarning = (ImageView) findViewById(R.id.imgwarning);

        if(ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
        {
            ActivityCompat.requestPermissions(MainActivity.this,
                    new String[]{
                            Manifest.permission.CAMERA
                    }, 100);
        }

        imgwarning.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                DisplayMetrics displayMetrics = new DisplayMetrics();
                getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
                int height = displayMetrics.heightPixels/3;
                ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) cardWarning.getLayoutParams();
                if(layoutParams.height != (height*2))
                {
                    cardWarning.setVisibility(View.VISIBLE);
                    layoutParams.height = height*2;
                }
                else
                {
                    layoutParams.height = (height/10);
                    cardWarning.setVisibility(View.INVISIBLE);
                }
                layoutParams.width = ViewGroup.LayoutParams.MATCH_PARENT;
                cardWarning.setLayoutParams(layoutParams);
                infowarning.setText("Los datos proporcionados por la aplicación sobre los tratamientos con plantas medicinales están en base a estudios realizados por el Ministerio de Salud Pública y Asistencia Social\n\n" +
                        "Esta aplicación proporciona tratamientos para enfermedades comunes con síntomas leves, por lo que recomendamos que si los síntomas son frecuentes o prolongados se visite un especialista en la medicina.\n\n" +
                        "La aplicación aún se encuentra en su versión beta lo que puede generar errores en la identificación de plantas medicinales, que aún no estén contenidas en ella.\n\n");
            }
        });

        ejemplo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view){
                DisplayMetrics displayMetrics = new DisplayMetrics();
                getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
                int height = displayMetrics.heightPixels/3;
                ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) cardView.getLayoutParams();
                if(layoutParams.height != (height*2))
                {
                    layoutParams.height = height*2;
                }
                else
                {
                    layoutParams.height = (height/9);
                }
                layoutParams.width = ViewGroup.LayoutParams.MATCH_PARENT;
                cardView.setLayoutParams(layoutParams);
            }
        });

        imageView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view){
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent,100);
                /*Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent, "Seleccione una imagen"), 12);*/

                try{
                    tflite = new Interpreter(loadmodelfile(MainActivity.this));
                }catch (IOException e){
                    e.printStackTrace();
                }

                btnClassify.setOnClickListener(new View.OnClickListener(){
                    @Override
                    public void onClick(View view){
                        int imageTensorIndex = 0;
                        int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape();
                        imageSizeX = imageShape[1];
                        imageSizeY = imageShape[2];
                        DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();

                        int probabilityTensorIndex = 0;
                        int[] probabilityShape = tflite.getOutputTensor(probabilityTensorIndex).shape();
                        DataType probabilityDataType = tflite.getInputTensor(probabilityTensorIndex).dataType();

                        inputImageBuffer = new TensorImage();
                        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape,probabilityDataType);
                        probabilityProcessor = new TensorProcessor.Builder().add(getPostProcessorNormalizeOP()).build();

                        inputImageBuffer = loadImage(bitmap);
                        tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
                        showresults();
                    }
                });
            }
        });
    }

    private TensorImage loadImage(final Bitmap bitmap){
        inputImageBuffer.load(bitmap);

        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());

        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize,cropSize))
                        .add(new ResizeOp(imageSizeX,imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(getPreProcessorNormalizeOP())
                        .build();

        return imageProcessor.process(inputImageBuffer);
    }

    private MappedByteBuffer loadmodelfile(Activity activity) throws IOException{
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd("Plantas_model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startoffset = fileDescriptor.getStartOffset();
        long declareLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startoffset, declareLength);
    }

    private TensorOperator getPreProcessorNormalizeOP(){
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }

    private TensorOperator getPostProcessorNormalizeOP() {
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }

    public static void barChart(BarChart barChart, ArrayList<BarEntry> arrayList, final ArrayList<String> xAxisValues){
        barChart.setDrawBarShadow(false);
        barChart.setFitBars(true);
        barChart.setDrawValueAboveBar(true);
        barChart.setMaxVisibleValueCount(25);
        barChart.setPinchZoom(true);
        barChart.setDrawGridBackground(true);

        BarDataSet barDataSet = new BarDataSet(arrayList, "Clase");
        barDataSet.setColors(new int[]{Color.parseColor("#03A9F4"), Color.parseColor("#FF9800"),
                Color.parseColor("#76FF03"), Color.parseColor("#E91E63"), Color.parseColor("#2962FF")});

        BarData barData = new BarData(barDataSet);
        barData.setBarWidth(0.9f);
        barData.setValueTextSize(0f);

        barChart.setBackgroundColor(Color.WHITE);
        barChart.setDrawGridBackground(false);
        barChart.animateY(2000);

        XAxis xAxis = barChart.getXAxis();
        xAxis.setTextSize(13f);
        xAxis.setTextColor(Color.BLACK);
        xAxis.setPosition(XAxis.XAxisPosition.TOP_INSIDE);
        xAxis.setValueFormatter(new IndexAxisValueFormatter(xAxisValues));
        xAxis.setDrawGridLines(false);

        barChart.setData(barData);
    }

    private void showresults(){
        try{
            labels = FileUtil.loadLabels(MainActivity.this,"Plantas_labels.txt");
        }catch (IOException e){
            e.printStackTrace();
        }

        Map<String, Float> labelsProbability =
                new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
                .getMapWithFloatValue();
        float maxValueinMap = (Collections.max(labelsProbability.values()));

        for (Map.Entry<String, Float> entry: labelsProbability.entrySet()){
            String[] label = labelsProbability.keySet().toArray(new String[0]);
            Float[] label_probability = labelsProbability.values().toArray(new Float[0]);

            mBarChat = findViewById(R.id.chart);
            mBarChat.getXAxis().setDrawGridLines(false);
            mBarChat.getAxisLeft().setDrawGridLines(false);

            ArrayList<BarEntry> barEntries = new ArrayList<>();
            int x = 0;
            float ant = 0;

            for(int i = 0; i< label_probability.length; i++)
            {
                barEntries.add(new BarEntry(i,label_probability[i]*100));
                if(ant<(label_probability[i]*100)){
                    ant = label_probability[i]*100;
                    x = i;
                }
            }

            ArrayList<String> xAxisName = new ArrayList<>();
            for(int i=0; i<label.length; i++){
                xAxisName.add(label[i]);
            }

            barChart(mBarChat, barEntries, xAxisName);
            prediction.setText("Predicción");
            carga(x);
        }
    }

    public void carga(int x){
        Drawable myDrawable = getResources().getDrawable(R.drawable.apazote);

        switch (x)
        {
            case 0:
                myDrawable = getResources().getDrawable(R.drawable.apazote);
                titulo.setText("Apazote");
                cuerpo.setText("Descripción:\nHierba de fuerte olor fétido, ramosa, tallo acanalado, rojizo, 60-150 cm de alto. Hojas alternas, sin tallo, 2-9 cm de largo. \n\n" +
                        "Uso e indicaciones:\nUso en asmáticos y para la digestión. Es antiséptica, antifúngica, antiparasitaria, cicatrizante, colagoga, diurética, emenagoga, sudorífica y tónica. El aceite es hipotensor, antifúngico, relajante muscular y estimulante respiratorio.\n\n" +
                        "Dosis y tratamiento:\nBeber 1 taza de la planta hasta por 3 días; según lo OMS la dosis única de 20 g es efectiva o 2 cucharadas de jarabe hasta notar mejoras en el paciente.\n\n" +
                        "Contraindicaciones:\nEn pacientes debilitados, ancianos y embarazadas, usada contra varios parásitos, pero su dosis terapéutica es cercana a la dosis toxica, por lo que su uso puede ser cuidadoso y por tiempo limitado, consumir únicamente las hojas.\n");
                break;
            case 1:
                myDrawable = getResources().getDrawable(R.drawable.hierbabuena);
                titulo.setText("Hierbabuena");
                cuerpo.setText("Descripción:\nHierba aromática, tallo rastrero, cuadrangular, 1 m de alto. Hoja verde brillante, hojas con pequeños dientes en la orilla. \n\n" +
                        "Uso e indicaciones:\nEl cocimiento de hojas y planta se usa por vía oral para tratar afecciones digestivas (cólico, indigestión, diarrea, dispepsia, gases, gastralgia, nausea) dismenorrea, reumatismo y neuralgia. Uso externo o Tópicamente en cataplasma y compresas para tratar obsesos, reumatismo y tumores en baños para relajar y limpiar los malos olores en los pies, lavar heridas y raspones.\n\n" +
                        "Dosis y tratamiento:\nTomar de 2-3 veces/ día después de las comidas durante 3-4 semanas en dosis de:\n" +
                        "•\t1 cucharada por taza en infusión o te.\n" +
                        "•\t2-4 ml de tintura 1 vaso de agua.\n" +
                        "•\t15-20 gotas de extracto de aceite. \n" +
                        "Aplicar sobre la piel en preparaciones liquidas o semisólidas a base del extracto como tintura o pomada en el área afectada.\n\n" +
                        "Contraindicaciones:\nNo se ha reportado contraindicaciones de las hojas. La esencia está contraindicada en personas con hipersensibilidad individual o hernia diafragmática. En personas sensibles puede producir nerviosismo e insomnio, consumir únicamente las hojas de la planta.\n");
                break;
            case 2:
                myDrawable = getResources().getDrawable(R.drawable.manzanilla);
                titulo.setText("Manzanilla");
                cuerpo.setText("Descripción:\nEsta planta mide un aproximado de 60 cm de alto, posee una flor solitaria o agrupadas con pétalos blancos redondos y vilano en forma de corona. \n\n" +
                        "Uso e indicaciones:\nSon usadas para tratar diarrea, dispepsia, gases, gastralgia, gastritis, indigestión, para aumentar el apetito, inflamación urinaria, amigdalitis, cefalea, convulsiones, Dismenorrea, histeria, insomnio nerviosismo y reumatismo. \n\n" +
                        "Dosis y tratamientos:\n" +
                        "•\t10-20 gotas en tintura 1:8 en etanol 35%. \n" +
                        "•\t1-3 ml gotas de extracto fluido en 1:1 en etanol 45%. \n" +
                        "•\t1-3 g de jarabe. Aplicar tópicamente como compresa, loción, lavado, baño, colutorio, irrigación vaginal o anal. Administrar 3-4 veces/día durante 5-6 semanas en dosis de: 1-2 g/taza en infusión.\n\n" +
                        "Contraindicaciones:\nNo prescribir el aceite esencial durante el embarazo, ni en pacientes con gastritis, colitis y ùlcera péptica, no consumir otra parte que no sea la flor de la planta.\n");
                break;
            case 3:
                myDrawable = getResources().getDrawable(R.drawable.sabila);
                titulo.setText("Sabila");
                cuerpo.setText("Descripción:\nHojas largas y gruesas, 30-60 cm de largo, verde claro, márgenes con dientes espinosos; escapo robusto. \n\n" +
                        "Uso e indicaciones:\nCulturalmente el extracto y gel se usa oralmente para tratar acné, artritis, reumatismo y úlceras gástricas; la infusión para tratar y afecciones hepáticas. Uso externo se usa para tratar, irritación, psoriasis, quemaduras, raspones, úlceras, verrugas y cicatrizar heridas.\n\n" +
                        "Dosis y tratamiento:\nAdministrar una vez al día en ayunas por un máximo de 15 días en dosis de:\n" +
                        "•\t1 cucharada por taza en apagado.\n" +
                        "•\t2 a 4 mil de tintura en ½ vaso de agua.\n" +
                        "•\t1 a 3 cucharada de jarabe, 3 veces al día.\n" +
                        "Como laxante está indicada una sola dosis en la noche de 0.1 g/ día y como purgante 0.2-0.5 g/día. Aplicar tópicamente en crema, ungüento y otras formas de cosmética medicada.\n\n" +
                        "Contradicciones:\nNo usar durante el Embarazo, hemorroides, prostatitis y cistitis.\n");
                break;
            default:
                myDrawable = null;
                titulo.setText("");
                cuerpo.setText("");
                break;
        }

        try
        {
            if(x!=4)
                cardView.setVisibility(View.VISIBLE);
            else
                cardView.setVisibility(View.INVISIBLE);

            cuerpo.setMovementMethod(new ScrollingMovementMethod());
            ejemplo.setImageDrawable(myDrawable);
        }
        catch (Exception ex)
        {
            cardView.setVisibility(View.INVISIBLE);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        /*
        if(requestCode == 12 && resultCode == RESULT_OK && data != null){
            imageUri = data.getData();

            try{
                bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);
                imageView.setImageBitmap(bitmap);
            }catch (IOException e){
                e.printStackTrace();
            }
        }*/

        super.onActivityResult(requestCode, resultCode, data);
        try{
            if (requestCode == 100) {
                Bitmap capture = (Bitmap) data.getExtras().get("data");
                imageUri = data.getData();
                bitmap = capture;
                imageView.setImageBitmap(capture);
            }
        }
        catch (Exception ex)
        {

        }
    }
}