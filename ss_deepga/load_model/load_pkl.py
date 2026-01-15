"""
01
"""
import os
import pickle
import torch


from ss_deepga.resources.constants import EXECUTION_ID, CHECKPOINT_DIR

execution_id = EXECUTION_ID
final_epochs = 150
chck_dir = CHECKPOINT_DIR

pkl_path = os.path.join(chck_dir, f"Model_Exec_{execution_id}_Epoch_{final_epochs}_point.pkl")
print("Openning:", pkl_path)

with open(pkl_path, "rb") as f:
    values = pickle.load(f)

print("Llaves dentro del pkl:", list(values.keys()))

model = values["modelo"]
print("Tipo de modelo:", type(model))


# Guardar modelo completo (pickle-based)
full_pt_path = os.path.join(
    chck_dir, f"Model_Exec_{execution_id}_Epoch_{final_epochs}_full_model.pt"
)

model.eval()
model.cpu()

torch.save(model, full_pt_path)
print("Saved full model to:", full_pt_path)

# Probar que carga y que se puede usar
loaded_model = torch.load(full_pt_path, map_location="cpu", weights_only=False)
loaded_model.eval()
print("Loaded model type:", type(loaded_model))
print("OK: model loaded")
state_path = os.path.join(chck_dir, f"Model_Exec_{execution_id}_Epoch_{final_epochs}_state_dict.pt")
torch.save(model.state_dict(), state_path)
print("Saved state_dict to:", state_path)