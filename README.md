# Face swap detection (spatial + frequency fusion, Grad-CAM, cross-dataset test)

Novelty: (1) multi-branch model: EfficientNet-B0 on pixels + FFT-spectrum CNN and/or a fixed
SRM noise-residual CNN, (2) blended synthetic fakes made from real faces (Face X-ray / SBI style),
(3) trained on one dataset, tested on another, (4) Grad-CAM heatmap in a web app.

## 1. Setup
python -m venv venv
venv\Scripts\activate          (Windows)   |   source venv/bin/activate   (Linux/Mac)
pip install -r requirements.txt
# optional, better face detector:
pip install facenet-pytorch --no-deps

## 2. Dataset layout
Put videos like this:
  data_raw/ffpp/real/*.mp4      data_raw/ffpp/fake/*.mp4        (training dataset)
  data_raw/celebdf/real/*.mp4   data_raw/celebdf/fake/*.mp4     (cross-dataset test)

## 3. Extract face crops
python data_prep.py --src data_raw/ffpp    --out data/ffpp     (use 300+ videos per class for better accuracy)
python data_prep.py --src data_raw/celebdf --out data/celebdf --force-split test
# optional: add --png to save lossless crops (better for the SRM noise branch, uses more disk)

For a Kaggle image folder containing `real/` and `fake/` folders (frame filenames
such as `videoid_0.png`), create train/validation/test splits by source-video ID:
python split_image_dataset.py --src input_kaggle_dataset --out data/kaggle_faces

## 4. Train
python train.py --data data/ffpp          (original model: spatial + FFT; 15 epochs, early stopping on val AUC)
python train.py --data data/kaggle_faces --branch spatial --epochs 8 --batch-size 32 --workers 0 --image-size 128 --freeze-spatial --lr 0.001
python calibrate_threshold.py --data data/kaggle_faces --ckpt checkpoints/best.pt
python evaluate.py --data data/kaggle_faces/test --name kaggle_video_test

Recommended (new):
python train.py --data data/ffpp --branch srm --synthetic-ratio 0.5

Options:
  --branch spatial | fft | srm | fft+srm     which branches to use (default fft = original model)
  --synthetic-ratio R   add R x (#real FF++ fakes) blended fakes per epoch, built from real faces only
  --dual-view auto|on|off   FFT/SRM branches see a clean view (no JPEG/blur); auto = on for srm
  --no-freq             same as --branch spatial (old command still works)
Checkpoints store the branch, so evaluate.py / inference.py / app.py load the right model automatically.

## 5. Evaluate (same dataset, then different dataset)
python evaluate.py --data data/ffpp/val   --name ffpp_val
python evaluate.py --data data/celebdf/test --name celebdf_test
python evaluate.py --data data/celebdf/test --ckpt checkpoints/ablation_spatial.pt --name celebdf_test --roc-suffix _spatial

## 6. Run the web app
python app.py        ->  open http://127.0.0.1:5000

The bundled checkpoint/sample images are demonstrations, not proof of real-world
accuracy. The app only gives real/fake verdicts after validation on both classes
and separate held-out frame and source-video balanced accuracies of at least
0.75. Older,
initialized, or lower-performing checkpoints show an unverified result instead.
Scores are specific to their dataset and do not guarantee accuracy on passport
photos, webcams, or other unseen sources. There is no 100% reliable detector.
The current model trained from `input_kaggle_dataset` achieved 0.649 frame
balanced accuracy and 0.704 source-video top-30% balanced accuracy on the
source-video-disjoint test set, so it remains disabled for real/fake verdicts.
Its neural-network output is not a calibrated confidence percentage.

## 7. Ablation table for your report (one command)
python ablation.py              # trains + evaluates: spatial, spatial+FFT, spatial+SRM, spatial+SRM+synthetic
python ablation.py --only-report   # rebuild the table later from results/metrics.json
Other rows: --configs spatial fft srm srm_synth fft_srm fft_srm_synth   (table is saved to results/ablation.md)

| Model                             | FF++ val AUC | Celeb-DF AUC |
| Spatial only                      |              |              |
| Spatial + FFT (original)          |              |              |
| Spatial + SRM                     |              |              |
| Spatial + SRM + synthetic fakes   |              |              |

## Notes
- Synthetic fakes are used for TRAINING only. Validation and test sets are never modified.
- The blend mask is a deformed, feathered ellipse (no landmark detector needed). A landmark convex hull
  can be swapped into `_face_mask` in dataset.py if you want to try it.
- data_prep.py resizes crops to 224, which smooths fine noise a little; the SRM branch still works,
  but --png crops avoid the extra JPEG layer.

## 8. Extra tools (video timeline, robustness, ROC comparison)
Web app (python app.py): videos now use 16 frames, the verdict is the average of the most suspicious 30% of frames
(not the plain mean), a per-frame probability chart is shown, and scores within +/-0.1 of the threshold show "Uncertain".

python robustness.py --data data/celebdf/test --name celebdf_test --ckpt checkpoints/ablation_spatial.pt checkpoints/ablation_fft.pt checkpoints/ablation_srm.pt
    -> results/robustness_celebdf_test.csv / .png  (AUC under JPEG, blur, downscale, noise)

python evaluate.py ...     now also prints/stores EER, precision, recall and saves results/preds_<name><suffix>.npz
python compare_models.py --name celebdf_test
    -> results/roc_compare.png  (all evaluated models on one ROC plot; run python ablation.py first)
