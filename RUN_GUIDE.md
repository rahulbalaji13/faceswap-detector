# Run guide (line by line)

## 0. Project folder-ku ulle po
cd faceswap-detector

## 1. Virtual env + install
python -m venv venv
venv\Scripts\activate            (Windows)   |   source venv/bin/activate   (Linux/Mac)
pip install -r requirements.txt

## 2A. Dataset = VIDEOS (FF++ / Celeb-DF)
Videos-a inga podu:
  data_raw/ffpp/real/*.mp4     data_raw/ffpp/fake/*.mp4
  data_raw/celebdf/real/*.mp4  data_raw/celebdf/fake/*.mp4
python data_prep.py --src data_raw/ffpp    --out data/ffpp
python data_prep.py --src data_raw/celebdf --out data/celebdf --force-split test

## 2B. Dataset = IMAGES (Kaggle 140k real & fake faces)
python prepare_image_dataset.py --src "PATH/real_vs_fake/real-vs-fake" --out data/ffpp --limit 5000

## 3. Train
python train.py --data data/ffpp --branch srm --synthetic-ratio 0.5
(CPU-la slow na:  python train.py --data data/ffpp --branch spatial --epochs 5 --batch-size 16 --workers 0)

## 4. Evaluate
python evaluate.py --data data/ffpp/val --name ffpp_val
python evaluate.py --data data/celebdf/test --name celebdf_test     (celebdf iruntha mattum)

## Kaggle image dataset already in this project
python split_image_dataset.py --src input_kaggle_dataset --out data/kaggle_faces
python train.py --data data/kaggle_faces --branch spatial --epochs 6 --batch-size 32 --workers 0 --image-size 128 --freeze-spatial --lr 0.001
python calibrate_threshold.py --data data/kaggle_faces --ckpt checkpoints/best.pt
python evaluate.py --data data/kaggle_faces/test --ckpt checkpoints/best.pt --name kaggle_grouped_test

## 5. Web app
python app.py        -> browser-la http://127.0.0.1:5000

The sample checkpoint is not validated for real photos or webcam use. The app
shows an unverified result until you train with labeled real/fake faces and a
held-out validation split. Even then, accuracy is specific to the evaluated
dataset; no detector can guarantee 100% accuracy on unseen photos or cameras.
Both frame-level and grouped-video balanced accuracy must reach 75% on the
held-out test data. The current Kaggle run did not meet the quality gate, so the app
withholds real/fake verdicts rather than show misleading confidence percentages.

## 6. Optional
python ablation.py
python compare_models.py --name celebdf_test
