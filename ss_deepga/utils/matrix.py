import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, classification_report

from ss_deepga.resources.constants import EXECUTION_ID, CHECKPOINT_FULL_DIR, PATH_TO_CLASSES, FINAL_EPOCHS, IMAGE_SIZE
from ss_deepga.stratified_loaders import make_stratified_loaders_v2
# Importa función de loaders

def load_model_from_pkl(pkl_path, device):
    with open(pkl_path, "rb") as f:
        values = pickle.load(f)
    model = values["modelo"]
    model.eval()
    model.to(device)
    return model

def load_model_from_full_pt(full_pt_path, device):
    model = torch.load(full_pt_path, map_location=device, weights_only=False)
    model.eval()
    model.to(device)
    return model

@torch.no_grad()
def predict_all(model, dataloader, device):
    y_true = []
    y_pred = []

    for xb, yb in dataloader:
        xb = xb.to(device, dtype=torch.float32)
        yb = yb.to(device)
        if yb.dtype != torch.long:
            yb = yb.long()

        logits = model(xb) # (B, num_classes)
        preds = torch.argmax(logits, dim=1)

        y_true.append(yb.cpu().numpy())
        y_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return y_true, y_pred

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    execution_id = EXECUTION_ID
    final_epochs = FINAL_EPOCHS
    chck_dir = CHECKPOINT_FULL_DIR

    pkl_path = os.path.join(chck_dir, f"Model_Exec_{execution_id}_Epoch_{final_epochs}_point.pkl")
    full_pt_path = os.path.join(chck_dir, f"Model_Exec_{execution_id}_Epoch_{final_epochs}_full_model.pt")

    # 1) Carga dataset/test_dl (IMPORTANTE: mismo preprocessing que el entrenamiento)
    data_dir = PATH_TO_CLASSES  # <-- Ajusta data_dir ruta real
    train_dl, val_dl, test_dl, n_channels, n_classes, out_size, ds, train_idx, val_idx, test_idx = make_stratified_loaders_v2(
        data_dir=data_dir,
        image_size=IMAGE_SIZE,
        batch_size=32,
        val_split=0.15,
        test_split=0.15,
        seed=42,
        num_workers=2,
    )

    class_names = ds.classes
    print("Classes:", class_names)
    print("Test examples:", len(test_dl.dataset))

    # 2) Carga el modelo
    if os.path.exists(full_pt_path):
        print("Loading full model .pt:", full_pt_path)
        model = load_model_from_full_pt(full_pt_path, device)
    else:
        print("Loading model from .pkl:", pkl_path)
        model = load_model_from_pkl(pkl_path, device)

    # 3) Predicciones en test
    y_true, y_pred = predict_all(model, test_dl, device)

    # 4) Confusion matrix y reporte
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    out_png = os.path.join(chck_dir, f"confusion_matrix_exec_{execution_id}_ep_{final_epochs}.png")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, values_format="d", cmap="Blues", colorbar=True)

    ax.set_title(f"Confusion Matrix (Exec {execution_id}, Epoch {final_epochs})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print("Saved confusion matrix image to:", out_png)
    print("\nConfusion matrix (rows=true, cols=pred):\n", cm)

    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

if __name__ == "__main__":
    main()
