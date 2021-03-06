## Results

### CIFAR-10 (R20-BN-ReLU)
| Epoch | Pretrain Epoch | SWA LR | Train ACC / NLL / cNLL | Valid ACC / NLL / cNLL | Test ACC / NLL / cNLL  |
| :-:   | :-:            | :-:    | :-:                    | :-:                    | :-:                    |
| 200   | 200            | -      | 99.91 / 0.007 / 0.031  | 92.98 / 0.272 / 0.224  | 92.49 / 0.274 / 0.229  |
|       | 180            | 0.05   | 96.74 / 0.101 / 0.108  | 92.92 / 0.218 / 0.216  | 91.92 / 0.235 / 0.232  |
|       | 180            | 0.02   | 98.84 / 0.041 / 0.056  | 93.84 / 0.208 / 0.198  | 92.86 / 0.223 / 0.212  |
|       | 180            | 0.01   | 99.65 / 0.018 / 0.038  | 93.24 / 0.232 / 0.209  | 92.66 / 0.247 / 0.222  |
|       | 150            | 0.05   | 96.76 / 0.098 / 0.105  | 92.72 / 0.213 / 0.211  | 92.29 / 0.231 / 0.228  |
|       | 150            | 0.02   | 99.00 / 0.037 / 0.054  | 93.72 / 0.211 / 0.199  | 92.97 / 0.215 / 0.204  |
|       | 150            | 0.01   | 99.73 / 0.016 / 0.033  | 93.48 / 0.217 / 0.199  | 93.15 / 0.220 / 0.202  |
|       | 120            | 0.05   | 96.52 / 0.105 / 0.111  | 92.98 / 0.218 / 0.217  | 92.12 / 0.232 / 0.230  |
|       | 120            | 0.02   | 98.78 / 0.043 / 0.059  | 93.18 / 0.213 / 0.204  | 92.59 / 0.230 / 0.218  |
|       | 120            | 0.01   | 99.75 / 0.015 / 0.035  | 93.32 / 0.223 / 0.200  | 92.83 / 0.236 / 0.212  |
|       | 100            | 0.05   | 96.54 / 0.105 / 0.112  | 92.90 / 0.215 / 0.213  | 92.10 / 0.232 / 0.229  |
|       | 100            | 0.02   | 98.79 / 0.043 / 0.058  | 93.34 / 0.210 / 0.200  | 92.75 / 0.228 / 0.216  |
|       | 100            | 0.01   | 99.68 / 0.017 / 0.038  | 92.84 / 0.235 / 0.211  | 92.67 / 0.241 / 0.217  |
| 400   | 400            | -      | 99.98 / 0.004 / 0.022  | 93.34 / 0.266 / 0.220  | 92.79 / 0.273 / 0.228  |
|       | 360            | 0.05   | 96.98 / 0.094 / 0.099  | 93.16 / 0.205 / 0.204  | 92.18 / 0.225 / 0.222  |
|       | 360            | 0.02   | 99.28 / 0.029 / 0.043  | 93.84 / 0.196 / 0.187  | 93.23 / 0.212 / 0.200  |
|       | 360            | 0.01   | 99.78 / 0.012 / 0.030  | 93.76 / 0.222 / 0.198  | 92.83 / 0.233 / 0.207  |
|       | 300            | 0.05   | 97.14 / 0.089 / 0.096  | 93.04 / 0.205 / 0.203  | 92.64 / 0.221 / 0.218  |
|       | 300            | 0.02   | 99.24 / 0.030 / 0.045  | 93.78 / 0.201 / 0.191  | 93.08 / 0.211 / 0.200  |
|       | 300            | 0.01   | 99.88 / 0.010 / 0.026  | 93.50 / 0.214 / 0.191  | 93.52 / 0.227 / 0.201  |
|       | 240            | 0.05   | 96.91 / 0.094 / 0.101  | 93.24 / 0.210 / 0.208  | 92.62 / 0.227 / 0.224  |
|       | 240            | 0.02   | 99.25 / 0.031 / 0.045  | 93.72 / 0.201 / 0.190  | 93.23 / 0.220 / 0.207  |
|       | 240            | 0.01   | 99.82 / 0.013 / 0.032  | 93.56 / 0.225 / 0.200  | 93.32 / 0.229 / 0.206  |
|       | 200            | 0.05   | 96.72 / 0.102 / 0.108  | 93.02 / 0.215 / 0.214  | 92.30 / 0.231 / 0.229  |
|       | 200            | 0.02   | 99.16 / 0.033 / 0.048  | 93.70 / 0.204 / 0.193  | 93.00 / 0.217 / 0.204  |
|       | 200            | 0.01   | 99.85 / 0.011 / 0.028  | 93.88 / 0.214 / 0.191  | 93.09 / 0.231 / 0.206  |
| 600   | 600            | -      | 99.98 / 0.003 / 0.020  | 93.56 / 0.259 / 0.214  | 92.55 / 0.284 / 0.235  |
|       | 540            | 0.05   | 97.23 / 0.085 / 0.093  | 93.38 / 0.201 / 0.199  | 92.66 / 0.217 / 0.214  |
|       | 540            | 0.02   | 99.38 / 0.026 / 0.041  | 94.12 / 0.200 / 0.188  | 93.32 / 0.212 / 0.199  |
|       | 540            | 0.01   | 99.89 / 0.009 / 0.025  | 94.18 / 0.211 / 0.188  | 93.33 / 0.227 / 0.203  |
|       | 450            | 0.05   | 97.26 / 0.087 / 0.095  | 93.10 / 0.209 / 0.206  | 92.48 / 0.217 / 0.215  |
|       | 450            | 0.02   | 99.38 / 0.027 / 0.041  | 93.82 / 0.201 / 0.191  | 93.57 / 0.211 / 0.199  |
|       | 450            | 0.01   | 99.90 / 0.009 / 0.026  | 93.74 / 0.222 / 0.196  | 93.32 / 0.230 / 0.204  |
|       | 360            | 0.05   | 97.10 / 0.088 / 0.095  | 93.16 / 0.201 / 0.199  | 92.79 / 0.218 / 0.214  |
|       | 360            | 0.02   | 99.34 / 0.026 / 0.043  | 93.98 / 0.203 / 0.189  | 93.27 / 0.213 / 0.201  |
|       | 360            | 0.01   | 99.92 / 0.008 / 0.024  | 93.80 / 0.217 / 0.191  | 93.13 / 0.228 / 0.200  |
|       | 300            | 0.05   | 97.08 / 0.089 / 0.095  | 93.54 / 0.202 / 0.200  | 92.33 / 0.213 / 0.211  |
|       | 300            | 0.02   | 99.35 / 0.028 / 0.041  | 93.80 / 0.188 / 0.180  | 93.66 / 0.199 / 0.191  |
|       | 300            | 0.01   | 99.89 / 0.009 / 0.024  | 94.12 / 0.205 / 0.183  | 93.27 / 0.227 / 0.201  |

### CIFAR-100 (R20-BN-ReLU)
| Epoch | Pretrain Epoch | SWA LR | Train ACC / NLL / cNLL | Valid ACC / NLL / cNLL | Test ACC / NLL / cNLL  |
| :-:   | :-:            | :-:    | :-:                    | :-:                    | :-:                    |
| 200   | 200            | -      | 92.42 / 0.274 / 0.385  | 68.84 / 1.210 / 1.121  | 68.22 / 1.228 / 1.136  |
|       | 180            | 0.05   | 80.78 / 0.644 / 0.670  | 68.74 / 1.088 / 1.077  | 68.96 / 1.077 / 1.068  |
|       | 180            | 0.02   | 86.45 / 0.454 / 0.510  | 69.64 / 1.086 / 1.053  | 69.64 / 1.080 / 1.049  |
|       | 180            | 0.01   | 90.24 / 0.339 / 0.418  | 69.40 / 1.120 / 1.069  | 69.31 / 1.136 / 1.080  |
|       | 150            | 0.05   | 80.78 / 0.649 / 0.675  | 68.58 / 1.084 / 1.074  | 68.79 / 1.068 / 1.060  |
|       | 150            | 0.02   | 86.91 / 0.442 / 0.502  | 69.64 / 1.093 / 1.059  | 69.36 / 1.074 / 1.045  |
|       | 150            | 0.01   | 89.80 / 0.350 / 0.429  | 69.44 / 1.108 / 1.055  | 69.67 / 1.113 / 1.061  |
|       | 120            | 0.05   | 80.25 / 0.661 / 0.687  | 68.38 / 1.087 / 1.076  | 68.38 / 1.096 / 1.084  |
|       | 120            | 0.02   | 86.13 / 0.463 / 0.522  | 69.26 / 1.104 / 1.068  | 69.40 / 1.087 / 1.056  |
|       | 120            | 0.01   | 89.85 / 0.346 / 0.430  | 69.66 / 1.128 / 1.070  | 69.92 / 1.117 / 1.063  |
|       | 100            | 0.05   | 79.98 / 0.674 / 0.702  | 68.64 / 1.103 / 1.091  | 68.69 / 1.085 / 1.076  |
|       | 100            | 0.02   | 85.79 / 0.476 / 0.531  | 69.40 / 1.101 / 1.069  | 69.02 / 1.096 / 1.064  |
|       | 100            | 0.01   | 89.54 / 0.357 / 0.442  | 69.30 / 1.137 / 1.077  | 69.69 / 1.114 / 1.064  |
| 400   | 400            | -      | 95.70 / 0.175 / 0.306  | 68.40 / 1.270 / 1.144  | 68.15 / 1.276 / 1.147  |
|       | 360            | 0.05   | 82.05 / 0.608 / 0.635  | 69.34 / 1.059 / 1.048  | 69.97 / 1.047 / 1.036  |
|       | 360            | 0.02   | 88.33 / 0.395 / 0.462  | 70.36 / 1.084 / 1.041  | 70.40 / 1.065 / 1.028  |
|       | 360            | 0.01   | 91.92 / 0.285 / 0.372  | 69.62 / 1.135 / 1.071  | 69.73 / 1.130 / 1.065  |
|       | 300            | 0.05   | 81.60 / 0.617 / 0.645  | 69.70 / 1.062 / 1.050  | 69.01 / 1.073 / 1.061  |
|       | 300            | 0.02   | 88.48 / 0.392 / 0.454  | 70.42 / 1.068 / 1.031  | 70.36 / 1.052 / 1.019  |
|       | 300            | 0.01   | 91.81 / 0.286 / 0.373  | 69.62 / 1.119 / 1.055  | 69.79 / 1.130 / 1.065  |
|       | 240            | 0.05   | 81.59 / 0.622 / 0.648  | 69.70 / 1.066 / 1.056  | 69.45 / 1.051 / 1.043  |
|       | 240            | 0.02   | 87.04 / 0.431 / 0.493  | 69.20 / 1.091 / 1.054  | 69.89 / 1.089 / 1.055  |
|       | 240            | 0.01   | 91.96 / 0.285 / 0.377  | 69.48 / 1.133 / 1.065  | 70.26 / 1.106 / 1.046  |
|       | 200            | 0.05   | 81.03 / 0.635 / 0.659  | 69.48 / 1.065 / 1.055  | 69.09 / 1.075 / 1.064  |
|       | 200            | 0.02   | 87.57 / 0.418 / 0.472  | 70.70 / 1.047 / 1.017  | 69.88 / 1.069 / 1.036  |
|       | 200            | 0.01   | 90.15 / 0.341 / 0.417  | 70.02 / 1.102 / 1.054  | 69.28 / 1.130 / 1.075  |
| 600   | 600            | -      | 96.97 / 0.136 / 0.275  | 67.82 / 1.315 / 1.166  | 68.04 / 1.313 / 1.163  |
|       | 540            | 0.05   | 81.87 / 0.603 / 0.633  | 69.86 / 1.063 / 1.050  | 69.21 / 1.060 / 1.048  |
|       | 540            | 0.02   | 89.14 / 0.373 / 0.440  | 70.80 / 1.071 / 1.029  | 70.38 / 1.065 / 1.028  |
|       | 540            | 0.01   | 92.88 / 0.256 / 0.343  | 70.30 / 1.106 / 1.041  | 70.16 / 1.095 / 1.034  |
|       | 450            | 0.05   | 82.09 / 0.605 / 0.633  | 70.40 / 1.050 / 1.039  | 69.51 / 1.045 / 1.036  |
|       | 450            | 0.02   | 88.89 / 0.380 / 0.444  | 70.24 / 1.076 / 1.038  | 70.78 / 1.043 / 1.013  |
|       | 450            | 0.01   | 93.00 / 0.250 / 0.343  | 70.10 / 1.122 / 1.049  | 69.80 / 1.135 / 1.057  |
|       | 360            | 0.05   | 81.78 / 0.610 / 0.640  | 70.02 / 1.062 / 1.049  | 69.63 / 1.057 / 1.045  |
|       | 360            | 0.02   | 88.83 / 0.383 / 0.443  | 70.78 / 1.057 / 1.024  | 70.50 / 1.061 / 1.026  |
|       | 360            | 0.01   | 92.64 / 0.265 / 0.354  | 70.20 / 1.102 / 1.038  | 70.51 / 1.095 / 1.036  |
|       | 300            | 0.05   | 81.83 / 0.613 / 0.639  | 69.86 / 1.054 / 1.043  | 69.67 / 1.052 / 1.042  |
|       | 300            | 0.02   | 88.81 / 0.382 / 0.443  | 70.98 / 1.058 / 1.022  | 70.53 / 1.052 / 1.018  |
|       | 300            | 0.01   | 92.44 / 0.267 / 0.365  | 69.66 / 1.155 / 1.077  | 69.79 / 1.112 / 1.047  |

### CIFAR-10 (WRN28x10-BN-ReLU)
| Epoch | Pretrain Epoch | SWA LR | Train ACC / NLL / cNLL | Valid ACC / NLL / cNLL | Test ACC / NLL / cNLL  |
| :-:   | :-:            | :-:    | :-:                    | :-:                    | :-:                    |
| 200   | 200            | -      | 100.0 / 0.000 / 0.010  | 96.18 / 0.177 / 0.146  | 96.08 / 0.170 / 0.142  |
|       | 180            | 0.05   | 99.89 / 0.006 / 0.011  | 96.52 / 0.112 / 0.107  | 96.22 / 0.125 / 0.118  |
|       | 150            | 0.05   | 99.88 / 0.007 / 0.012  | 96.56 / 0.113 / 0.108  | 96.12 / 0.120 / 0.114  |
|       | 120            | 0.05   | 99.90 / 0.006 / 0.012  | 96.48 / 0.114 / 0.108  | 96.14 / 0.118 / 0.112  |
|       | 100            | 0.05   | 99.90 / 0.007 / 0.012  | 96.60 / 0.110 / 0.105  | 96.24 / 0.117 / 0.111  |
| 300   | 300            | -      | 100.0 / 0.001 / 0.011  | 96.32 / 0.168 / 0.143  | 96.40 / 0.163 / 0.140  |
|       | 270            | 0.05   | 99.90 / 0.006 / 0.011  | 96.40 / 0.112 / 0.106  | 96.35 / 0.118 / 0.112  |
|       | 225            | 0.05   | 99.91 / 0.006 / 0.011  | 96.66 / 0.110 / 0.104  | 96.19 / 0.119 / 0.114  |
|       | 180            | 0.05   | 99.91 / 0.005 / 0.011  | 96.72 / 0.110 / 0.104  | 96.30 / 0.119 / 0.112  |
|       | 150            | 0.05   | 99.91 / 0.006 / 0.011  | 96.74 / 0.111 / 0.104  | 96.12 / 0.119 / 0.112  |

### CIFAR-100 (WRN28x10-BN-ReLU)
| Epoch | Pretrain Epoch | SWA LR | Train ACC / NLL / cNLL | Valid ACC / NLL / cNLL | Test ACC / NLL / cNLL  |
| :-:   | :-:            | :-:    | :-:                    | :-:                    | :-:                    |
| 200   | 200            | -      | 99.98 / 0.001 / 0.004  | 80.04 / 0.900 / 0.844  | 80.63 / 0.843 / 0.802  |
|       | 180            | 0.05   | 99.68 / 0.011 / 0.026  | 80.62 / 0.803 / 0.679  | 81.08 / 0.785 / 0.669  |
|       | 150            | 0.05   | 99.68 / 0.012 / 0.028  | 81.60 / 0.806 / 0.672  | 81.49 / 0.777 / 0.659  |
|       | 120            | 0.05   | 99.60 / 0.013 / 0.031  | 81.06 / 0.796 / 0.673  | 81.29 / 0.769 / 0.656  |
|       | 100            | 0.05   | 99.43 / 0.018 / 0.038  | 81.12 / 0.775 / 0.665  | 81.24 / 0.759 / 0.658  |
| 300   | 300            | -      | 99.99 / 0.001 / 0.003  | 80.40 / 0.878 / 0.842  | 80.38 / 0.861 / 0.830  |
|       | 270            | 0.05   | 99.78 / 0.008 / 0.020  | 81.02 / 0.805 / 0.676  | 81.44 / 0.791 / 0.667  |
|       | 225            | 0.05   | 99.78 / 0.008 / 0.022  | 81.30 / 0.815 / 0.677  | 81.86 / 0.777 / 0.656  |
|       | 180            | 0.05   | 99.68 / 0.010 / 0.024  | 81.02 / 0.807 / 0.671  | 81.57 / 0.792 / 0.664  |
|       | 150            | 0.05   | 99.66 / 0.011 / 0.028  | 81.66 / 0.801 / 0.670  | 81.39 / 0.776 / 0.658  |

### TinyImageNet-200 (R18-BN-ReLU)
| Epoch | Pretrain Epoch | SWA LR | Train ACC / NLL / cNLL | Valid ACC / NLL / cNLL | Test ACC / NLL / cNLL  |
| :-:   | :-:            | :-:    | :-:                    | :-:                    | :-:                    |
| 100   | 100            | -      | 99.98 / 0.005 / 0.022  | 66.07 / 1.544 / 1.459  | 65.52 / 1.581 / 1.486  |
|       |  90            | 0.05   | 88.22 / 0.430 / 0.494  | 67.33 / 1.369 / 1.306  | 66.99 / 1.412 / 1.340  |
|       |  90            | 0.02   | 99.19 / 0.042 / 0.108  | 67.28 / 1.523 / 1.360  | 66.49 / 1.548 / 1.375  |
|       |  90            | 0.01   | 99.94 / 0.008 / 0.040  | 66.47 / 1.575 / 1.415  | 65.41 / 1.640 / 1.464  |
|       |  75            | 0.05   | 88.39 / 0.421 / 0.488  | 67.88 / 1.367 / 1.295  | 67.19 / 1.391 / 1.315  |
|       |  75            | 0.02   | 99.36 / 0.036 / 0.100  | 67.62 / 1.503 / 1.337  | 67.21 / 1.546 / 1.364  |
|       |  75            | 0.01   | 99.96 / 0.005 / 0.030  | 67.05 / 1.547 / 1.378  | 66.49 / 1.613 / 1.420  |
|       |  60            | 0.05   | 88.05 / 0.433 / 0.500  | 67.72 / 1.376 / 1.302  | 66.92 / 1.410 / 1.329  |
|       |  60            | 0.02   | 99.27 / 0.039 / 0.104  | 67.76 / 1.497 / 1.330  | 67.08 / 1.544 / 1.360  |
|       |  60            | 0.01   | 99.94 / 0.007 / 0.043  | 66.60 / 1.577 / 1.400  | 66.48 / 1.608 / 1.419  |
|       |  50            | 0.05   | 87.59 / 0.449 / 0.514  | 67.41 / 1.374 / 1.303  | 67.32 / 1.409 / 1.330  |
|       |  50            | 0.02   | 99.22 / 0.041 / 0.111  | 67.59 / 1.511 / 1.335  | 66.90 / 1.537 / 1.353  |
|       |  50            | 0.01   | 99.96 / 0.005 / 0.034  | 66.81 / 1.567 / 1.383  | 66.46 / 1.620 / 1.414  |
| 200   | 200            | -      | 99.98 / 0.002 / 0.007  | 66.57 / 1.529 / 1.489  | 65.12 / 1.585 / 1.535  |
|       | 180            | 0.05   | 90.18 / 0.357 / 0.427  | 68.58 / 1.360 / 1.280  | 67.92 / 1.380 / 1.299  |
|       | 180            | 0.02   | 99.71 / 0.018 / 0.063  | 67.72 / 1.522 / 1.344  | 67.76 / 1.543 / 1.355  |
|       | 180            | 0.01   | 99.97 / 0.002 / 0.017  | 67.31 / 1.580 / 1.402  | 66.57 / 1.605 / 1.418  |
|       | 150            | 0.05   | 89.86 / 0.365 / 0.431  | 68.86 / 1.346 / 1.267  | 68.43 / 1.363 / 1.281  |
|       | 150            | 0.02   | 99.76 / 0.014 / 0.057  | 67.98 / 1.519 / 1.324  | 67.47 / 1.575 / 1.357  |
|       | 150            | 0.01   | 99.98 / 0.002 / 0.014  | 67.75 / 1.560 / 1.382  | 66.79 / 1.611 / 1.414  |
|       | 120            | 0.05   | 89.70 / 0.372 / 0.441  | 68.55 / 1.354 / 1.271  | 67.50 / 1.386 / 1.295  |
|       | 120            | 0.02   | 99.75 / 0.015 / 0.062  | 68.04 / 1.546 / 1.334  | 67.43 / 1.572 / 1.348  |
|       | 120            | 0.01   | 99.98 / 0.002 / 0.017  | 67.95 / 1.571 / 1.379  | 67.15 / 1.589 / 1.390  |
|       | 100            | 0.05   | 89.48 / 0.382 / 0.450  | 68.48 / 1.354 / 1.271  | 68.02 / 1.385 / 1.295  |
|       | 100            | 0.02   | 99.72 / 0.018 / 0.067  | 67.99 / 1.518 / 1.322  | 67.62 / 1.556 / 1.345  |
|       | 100            | 0.01   | 99.98 / 0.002 / 0.017  | 67.82 / 1.581 / 1.379  | 67.48 / 1.613 / 1.402  |
| 400   | 400            | -      | 99.98 / 0.001 / 0.002  | 66.52 / 1.562 / 1.555  | 65.95 / 1.593 / 1.583  |
|       | 360            | 0.05   | 90.95 / 0.326 / 0.395  | 68.76 / 1.341 / 1.254  | 68.22 / 1.381 / 1.284  |
|       | 360            | 0.02   | 99.84 / 0.009 / 0.041  | 68.22 / 1.545 / 1.334  | 68.16 / 1.564 / 1.347  |
|       | 360            | 0.01   | 99.97 / 0.001 / 0.009  | 67.06 / 1.615 / 1.423  | 67.10 / 1.630 / 1.426  |
|       | 300            | 0.05   | 90.79 / 0.331 / 0.401  | 69.00 / 1.347 / 1.257  | 68.51 / 1.372 / 1.280  |
|       | 300            | 0.02   | 99.87 / 0.009 / 0.042  | 68.69 / 1.542 / 1.322  | 67.66 / 1.582 / 1.345  |
|       | 300            | 0.01   | 99.98 / 0.001 / 0.008  | 67.99 / 1.601 / 1.400  | 67.45 / 1.650 / 1.427  |
|       | 240            | 0.05   | 90.47 / 0.344 / 0.414  | 68.83 / 1.348 / 1.254  | 68.30 / 1.368 / 1.269  |
|       | 240            | 0.02   | 99.84 / 0.010 / 0.045  | 68.66 / 1.549 / 1.326  | 67.94 / 1.576 / 1.339  |
|       | 240            | 0.01   | 99.98 / 0.001 / 0.009  | 68.01 / 1.594 / 1.391  | 67.38 / 1.638 / 1.419  |
|       | 200            | 0.05   | 90.27 / 0.351 / 0.421  | 69.24 / 1.346 / 1.258  | 68.17 / 1.381 / 1.284  |
|       | 200            | 0.02   | 99.81 / 0.012 / 0.050  | 68.59 / 1.538 / 1.325  | 68.12 / 1.568 / 1.342  |
|       | 200            | 0.01   | 99.98 / 0.001 / 0.011  | 67.93 / 1.601 / 1.388  | 67.47 / 1.626 / 1.405  |
