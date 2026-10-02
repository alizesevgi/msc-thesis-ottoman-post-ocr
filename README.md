# Post-OCR Text Correction for Ottoman Turkish Transcription

Alize Sevgi Yalçınkaya. M.Sc. thesis, Computer Science and Engineering, Sabancı University, 2026.

**[Read the thesis (PDF)](Yalcinkaya_2026_MSc_Thesis.pdf)**

Official record: YÖK Ulusal Tez Merkezi, Tez No. 1028725. YÖK does not give theses a direct link. To find it there, enter the number in the Tez No field at https://tez.yok.gov.tr/UlusalTezMerkezi/tarama.jsp.

This repository contains only the thesis PDF.

## Abstract

Historical document digitization requires effective post-OCR error correction, particularly for low-resource scripts like Ottoman Turkish that lack modern native speakers and linguistic tools. We develop and evaluate fully automatic post-processing methods for Ottoman texts digitized via a frozen TrOCR-based recognizer, AKİS. Our corpus comprises 102,000 tokens across four books spanning 1732–1920, with token error rates of 17.3%–34.8%. We evaluate a single-stage ByT5 byte-level encoder-decoder for direct correction, and three error detection methods, to test whether detection could guide correction in a two-stage approach. The corrector is trained on synthetically augmented OCR–reference pairs, with one variant weighting the training objective toward low-confidence sequences. Both ByT5 variants achieve approximately 9.6% CER on the test set, an 8.07–8.55% relative reduction from the 10.51% OCR baseline. An error-type decomposition of the 3,092 test-set errors shows that the model copies 85% of errors unchanged and corrects circumflex substitutions at 42%, the only subtype with clear net benefit. Confidence-weighted training does not improve corpus-level CER but reduces overcorrections by 19%. In the two-stage approach, calibrated confidence thresholding achieves token-level F1 = 0.577, BERT-based classification reaches F1 = 0.640, and confidence-fused BERT reaches F1 = 0.647 (beta-calibrated concatenation fusion; 0.643 with raw confidence). All three produce well-calibrated probabilities (ECE 0.022–0.067). A quality-estimation experiment is negative: detector scores do not predict whether a correction will help or harm a sequence. These findings show that OCR confidence is most useful as a training-time risk-control signal rather than as an inference-time selector.

## Citation

```bibtex
@mastersthesis{yalcinkaya2026postocr,
  author = {Yal{\c{c}}{\i}nkaya, Alize Sevgi},
  title  = {Post-{OCR} Text Correction for {O}ttoman {T}urkish Transcription},
  school = {Sabanc{\i} University},
  year   = {2026},
  type   = {{M.Sc.} thesis},
  note   = {Y{\"O}K Ulusal Tez Merkezi, Tez No.\ 1028725}
}
```

© 2026 Alize Sevgi Yalçınkaya. All rights reserved.
