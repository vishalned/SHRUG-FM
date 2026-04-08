# SHRUG-FM: Reliability-Aware Foundation Models for Earth Observation

## Abstract
Geospatial foundation models (GFMs) for Earth observation often fail to perform reliably in environments underrepresented during pretraining. We introduce SHRUG-FM, a framework for reliability-aware prediction that enables GFMs to identify and abstain from likely failures. Our approach integrates three complementary signals: geophysical out-of-distribution (OOD) detection in the input space, OOD detection in the embedding space, and task-specific predictive uncertainty. We evaluate SHRUG-FM across three high-stakes rapid-mapping tasks: burn scar segmentation, flood mapping, and landslide detection. Our results show that SHRUG-FM consistently reduces prediction risk on retained samples, outperforming established single-signal baselines like predictive entropy. Crucially, by utilizing a shallow "glass-box" decision tree for signal fusion, SHRUG-FM provides interpretable abstention thresholds. It builds a pathway toward safer and more interpretable deployment of GFMs in climate-sensitive applications, bridging the gap between benchmark performance and real-world reliability.


## Cite
```bibtex
@misc{cohrs2025shrugfmreliabilityawarefoundationmodels,
      title={SHRUG-FM: Reliability-Aware Foundation Models for Earth Observation}, 
      author={Kai-Hendrik Cohrs and Zuzanna Osika and Maria Gonzalez-Calabuig and Vishal Nedungadi and Ruben Cartuyvels and Steffen Knoblauch and Joppe Massant and Shruti Nath and Patrick Ebel and Vasileios Sitokonstantinou},
      year={2025},
      eprint={2511.10370},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.10370}, 
}
```
