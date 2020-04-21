# Needletail: Scalable GPU-Accelerated Long Read Alignment
Needletail is a proof of concept long read alignment framework. Currently only global alignment is supported. We classify long reads as a query and target string greater than or equal to 1kbp. For reads less than 1kbp it was found that GASAL2 will provide substantially better performance (https://github.com/nahmedraja/GASAL2).

Needletail was designed as a term-project for EECS 570 - Parallel Computer Architecture, at the University of Michigan - Ann Arbor. We open source this project as a framework for future adaptations.
