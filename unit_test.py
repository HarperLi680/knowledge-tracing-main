import os

from Models.DSAKT import train_predict_DSAKT
from Models.ATKT import train_predict_ATKT
from Models.KTM import train_predict_KTM
from Models.bkt_bf import train_predict_BKT
from Models.PFA import train_predict_PFA
from Models.Elo import train_predict_Elo
from generate_predictions import calculate_n_skill

# Replace with actual minimal test files you have
TABULAR_DATA = "data/processed/train/tabular"
SEQUENTIAL_DATA = "data/processed/train/sequential"

def test_train_predict_BKT():
    files = sorted(os.listdir(TABULAR_DATA))
    preds, labels = train_predict_BKT(
        train_data=[os.path.join(TABULAR_DATA, files[0])],
        test_data=os.path.join(TABULAR_DATA, files[1])
    )
    assert len(preds) == len(labels), "BKT: Prediction-label length mismatch"

def test_train_predict_PFA():
    files = sorted(os.listdir(TABULAR_DATA))
    preds, labels = train_predict_PFA(
        train_data=[os.path.join(TABULAR_DATA, files[0])],
        test_data=os.path.join(TABULAR_DATA, files[1])
    )
    assert len(preds) == len(labels), "PFA: Prediction-label length mismatch"

def test_train_predict_KTM():
    files = sorted(os.listdir(TABULAR_DATA))
    preds, labels = train_predict_KTM(
        train_data=[os.path.join(TABULAR_DATA, files[0])],
        test_data=os.path.join(TABULAR_DATA, files[1])
    )
    assert len(preds) == len(labels), "KTM: Prediction-label length mismatch"

def test_train_predict_Elo():
    files = sorted(os.listdir(TABULAR_DATA))
    preds, labels = train_predict_Elo(
        train_files=[os.path.join(TABULAR_DATA, files[0])],
        test_file=os.path.join(TABULAR_DATA, files[1])
    )
    assert len(preds) == len(labels), "Elo: Prediction-label length mismatch"

def test_train_predict_DSAKT():
    files = sorted(os.listdir(SEQUENTIAL_DATA))
    preds, labels = train_predict_DSAKT(
        train_path=os.path.join(SEQUENTIAL_DATA, files[0]),
        valid_path=os.path.join(SEQUENTIAL_DATA, files[1]),
        test_path=os.path.join(SEQUENTIAL_DATA, files[2]),
        lr=0.3,
        window_size=50,
        dim=24,
        dropout=0.7,
        heads=8
    )
    assert len(preds) == len(labels), "DSAKT: Prediction-label length mismatch"

def test_train_predict_ATKT():
    files = sorted(os.listdir(SEQUENTIAL_DATA))
    all_files = [os.path.join(SEQUENTIAL_DATA, f) for f in files[:3]]
    n_skill = calculate_n_skill(all_files)
    preds, labels = train_predict_ATKT(
        train_data=[all_files[0]],
        valid_path=all_files[1],
        test_path=all_files[2],
        lr=0.001,
        gamma=0.5,
        lr_decay=50,
        hidden_emb_dim=80,
        skill_emb_dim=256,
        answer_emb_dim=96,
        beta=0.2,
        epsilon=10,
        n_skill=n_skill,
        seqlen=200,
        max_iter=1,
        batch_size=32
    )
    assert len(preds) == len(labels), "ATKT: Prediction-label length mismatch"


if __name__ == "__main__":
    test_train_predict_BKT()
    print("BKT test passed!")
    test_train_predict_PFA()
    print("PFA test passed!")
    test_train_predict_KTM()
    print("KTM test passed!")
    test_train_predict_Elo()
    print("Elo test passed!")
    test_train_predict_ATKT()
    print("ATKT test passed!")
    test_train_predict_DSAKT()
    print("DSAKT test passed!")
    print("All tests passed!")