import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data_processing import load_tabular_data, split_scale_tabular
from image_processing import create_generators, extract_cnn_features
from model import build_stacking_model
from evaluation import evaluate_model
from openai_report import generate_clinical_report
import tensorflow as tf 
import numpy as np
import joblib

if name == '__main__':
    X_tab, y_tab = load_tabular_data('data/medical1data.csv')
    X_train_tab, X_test_tab, y_train_tab, y_test_tab = split_scale_tabular(X_tab, y_tab)
    train_gen, val_gen, test_gen = create_generators('data/train', 'data/test')
    base_cnn = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base_cnn.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    cnn_model = tf.keras.models.Model(inputs=base_cnn.input, outputs=output)
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn_model.fit(train_gen, validation_data=val_gen, epochs=20)
    X_train_img, y_train_img = extract_cnn_features(train_gen, tf.keras.models.Model(inputs=base_cnn.input, outputs=x), train_gen.samples)
    X_test_img, y_test_img = extract_cnn_features(test_gen, tf.keras.models.Model(inputs=base_cnn.input, outputs=x), test_gen.samples)
    X_train_hybrid = np.hstack([X_train_tab, X_train_img])
    y_train_hybrid = y_train_tab
    stack_model = build_stacking_model()
    pipeline_hybrid = stack_model
    pipeline_hybrid.fit(X_train_hybrid, y_train_hybrid)
    X_test_hybrid = np.hstack([X_test_tab, X_test_img])
    y_test_hybrid = y_test_tab
    evaluate_model(pipeline_hybrid, X_test_hybrid, y_test_hybrid)
    joblib.dump(pipeline_hybrid, 'hybrid_cancer_model.pkl')
    patient_ids = list(range(len(y_test_hybrid)))
    predictions = pipeline_hybrid.predict(X_test_hybrid)
    clinical_report = generate_clinical_report(predictions, patient_ids)
    print(clinical_report)
