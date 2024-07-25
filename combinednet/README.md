# CombinedNet 
# 1. **To train the model:**
```bash
python train.py --data_path "datasetpath" --separate_val_path "validationpath" --separate_test_path "testpath" --metric mse --dataset_type regression --save_dir "checkpointpath" --target_columns label
```

# 2.**To take the inferrence:**
```bash
python predict.py --test_path "testdatapath" --checkpoint_dir "checkpointpath" --preds_path "predictionpath.csv"
```

# 3.**To train YOUR model:**

Your data should be in the format csv, and the column names are: 'smiles','label'.

You can freely tune the hyperparameter for your best performance (but highly recommend using the Bayesian optimization package).
