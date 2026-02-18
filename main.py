import argparse
from app import pipeline_hybrid, evaluate_model, generate_clinical_report
import joblib
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Brain Tumor MRI Hybrid Pipeline')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'predict'], help='Mode to run the pipeline')
    parser.add_argument('--test_patient_ids', type=int, nargs='+', help='Optional patient IDs for prediction report')
    args = parser.parse_args()

    if args.mode == 'train':
        print("Training the full hybrid pipeline...")
        # Training is handled inside app.py
        print("Training completed. Model saved as hybrid_cancer_model.pkl")
    
    elif args.mode == 'evaluate':
        print("Evaluating the model...")
        X_test_hybrid = pipeline_hybrid['X_test_hybrid']
        y_test_hybrid = pipeline_hybrid['y_test_hybrid']
        evaluate_model(pipeline_hybrid, X_test_hybrid, y_test_hybrid)

    elif args.mode == 'predict':
        if args.test_patient_ids:
            X_test_hybrid = pipeline_hybrid['X_test_hybrid']
            predictions = pipeline_hybrid.predict(X_test_hybrid)
            patient_ids = args.test_patient_ids
            report = generate_clinical_report(predictions, patient_ids)
            print(report)
        else:
            print("Please provide patient IDs using --test_patient_ids")

if name == '__main__':
    main()
