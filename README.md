# STGNN - Spatio-Temporal Graph Convolutional Network

A PyTorch-based implementation of Spatio-Temporal Graph Convolutional Networks (STGCN) for weather prediction using graph neural networks. This project leverages the relationships between geographically distributed weather stations to make accurate multi-step weather forecasts.

## 📋 Project Overview

STGNN applies spatio-temporal convolution to weather data across multiple stations. It models weather stations as nodes in a graph and captures spatial relationships through edges weighted by geodesic distances. The model predicts future weather conditions (temperature, humidity, precipitation) for the next 24 days using 60 days of historical data.

### Key Features

- **Spatio-Temporal Modeling**: Combines spatial relationships between weather stations with temporal patterns
- **Graph-Based Architecture**: Uses graph neural networks to capture dependencies between weather stations
- **Multi-Variable Prediction**: Predicts 7 weather parameters simultaneously
- **Geodesic Distance Integration**: Weights edges using actual geographical distances between stations
- **Normalized Data Handling**: Handles feature normalization and denormalization for numerical stability

## 🏗️ Architecture

### Model: STGCN_Best_Babu

The model consists of three STConv (Spatio-Temporal Convolution) blocks followed by fully connected layers:

```
Input (750 stations × 18 features × 60 timesteps)
    ↓
STConv Block 1: 18 → 32 → 64 channels (kernel size 9)
    ↓
STConv Block 2: 64 → 128 → 256 channels (kernel size 7)
    ↓
STConv Block 3: 256 → 128 → 64 channels (kernel size 5)
    ↓
Fully Connected Layer 1: 64 → 32
    ↓
Fully Connected Layer 2: 32 → 7 (output)
    ↓
Output (7 features × 24 timesteps per station)
```

### Input Features

The model processes 18 weather features for each station:

| Feature | Description |
|---------|-------------|
| T2M | Temperature at 2 meters |
| T2MWET | Wet bulb temperature |
| TS | Surface temperature |
| T2M_RANGE | Temperature range |
| T2M_MAX | Maximum temperature |
| T2M_MIN | Minimum temperature |
| QV2M | Specific humidity at 2m |
| RH2M | Relative humidity at 2m |
| PRECTOTCORR | Precipitation corrected |
| PS | Surface pressure |
| WS10M | Wind speed at 10m |
| WS10M_MAX | Max wind speed at 10m |
| WS10M_MIN | Min wind speed at 10m |
| WS10M_RANGE | Wind speed range at 10m |
| WS50M | Wind speed at 50m |
| WS50M_MAX | Max wind speed at 50m |
| WS50M_MIN | Min wind speed at 50m |
| WS50M_RANGE | Wind speed range at 50m |

### Output Features

The model predicts 7 features over 24 days:
- T2M_MIN (Minimum Temperature)
- RH2M (Relative Humidity)
- PRECTOTCORR (Precipitation)
- (4 additional features from the 18 input features)

## 📁 File Structure

```
STGNN/
├── Main_Test.py                          # Main testing and inference script
├── Graph Theory Project Analysis.pdf      # Project analysis documentation
├── STGNN Intro.pdf                       # Introduction and methodology
├── README.md                             # Documentation
└── .git/                                 # Version control
```

## 🔧 Core Components

### 1. **Geodesic Distance Calculator**

```python
def geodesic_distance(lat1, lon1, lat2, lon2)
```

Calculates the great-circle distance between two geographic coordinates using the Vincenty formula via the `geopy` library. Used to compute edge weights in the spatial graph.

### 2. **WeatherDatasetLoader**

```python
class WeatherDatasetLoader(object)
```

Prepares weather data for the model:
- **Inputs**: Time series snapshots, graph edges, and edge weights
- **Processing**: 
  - Extracts `lags` timesteps as features
  - Creates prediction targets from the last `pred_seq` timesteps
  - Organizes data into StaticGraphTemporalSignal format
- **Parameters**:
  - `lags` (int): Number of historical timesteps (default: 60)
  - `pred_days` (int): Number of days to predict (default: 24)

### 3. **Data Preprocessing Functions**

#### `normalizeTestData(test_file_path, mean_file_path, std_file_path)`
Normalizes test features using pre-computed mean and standard deviation values from the training set. Prevents data leakage and ensures numerical stability.

#### `get_features(df, stations)`
Extracts and organizes weather features for each station from the dataset. Returns a list of feature arrays sorted by station location.

#### `get_stations(filename)`
Loads station names/identifiers from a text file.

## 🚀 Usage

### Running the Model

```python
python Main_Test.py
```

### Configuration

Update the file paths in the `__main__` block:

```python
test_data_path = 'path/to/test_data.csv'
stations_path = 'path/to/stations.txt'
weights_path = 'path/to/model_weights.pt'
edge_index_path = 'path/to/edge_index.pt'
edge_weight_path = 'path/to/edge_weights.pt'
mean_file_path = 'path/to/mean_values.csv'
std_file_path = 'path/to/std_values.csv'
```

### Process Flow

1. **Load Data**: Reads test data and station information
2. **Normalize**: Applies feature normalization using pre-computed statistics
3. **Prepare Features**: Extracts weather features and organizes them by station
4. **Create Graph Dataset**: Builds graph structure with edge indices and weights
5. **Model Inference**: Loads pre-trained weights and runs predictions
6. **Denormalize**: Converts predictions back to original feature scale
7. **Evaluation**: Computes metrics (MSE, MAE, RMSE, R²)
8. **Visualization**: Plots predictions vs. ground truth for analysis

## 📊 Model Evaluation

The script computes standard regression metrics:

- **MSE** (Mean Squared Error): Average squared prediction error
- **MAE** (Mean Absolute Error): Average absolute prediction error  
- **RMSE** (Root Mean Squared Error): Square root of MSE
- **R²** (R-squared): Coefficient of determination

Per-station RMSE is also calculated to identify which stations have better/worse predictions.

## 🔗 Dependencies

- **PyTorch**: Deep learning framework
- **torch_geometric_temporal**: Spatio-temporal graph neural network layers
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **scikit-learn**: Machine learning metrics
- **geopy**: Geographical distance calculations
- **tqdm**: Progress bars
- **Matplotlib**: Visualization

## 📈 Expected Output

The script generates:
1. **Metrics**: MSE, MAE, RMSE, R² scores
2. **Sample Plots**:
   - Time series comparison for one station (predictions vs. ground truth)
   - Per-station RMSE distribution across all 750 stations
3. **Prediction Data**: Predictions for all 750 stations over 24 days

## 💡 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lags` | 60 | Historical timesteps for input |
| `pred_days` | 24 | Prediction horizon (days) |
| Kernel Sizes | 9, 7, 5 | STConv kernel sizes per block |
| Channels | [18→32→64], [64→128→256], [256→128→64] | Feature channels per block |
| Num Stations | 750 | Number of weather stations in the graph |

## 🛠️ Model Details

### STConv Blocks
Each STConv block applies:
- **Spatial convolution**: Graph convolutions across the station network
- **Temporal convolution**: 1D convolutions along the time dimension
- **Non-linearity**: Activation functions for feature extraction

### Prediction Horizon Calculation
For the 60-lag input:
- Prediction timesteps = Lags - 2×(Kernel - 1)×NumBlocks
- With kernels [9, 7, 5]: 60 - 2×(8 + 6 + 4) = **24 days**

## 📝 Notes

- The model expects exactly 60 timesteps of historical data
- Predictions for precipitation are clipped to 0 (no negative precipitation)
- Geodesic distance calculations enable location-aware spatial modeling
- Pre-trained weights must match the architecture specification

## 🔄 Future Enhancements

- Support for variable lag lengths
- Multi-step training with rolling predictions
- Attention mechanisms for dynamic edge weighting
- Ensemble methods for uncertainty quantification
- Real-time inference pipeline

## 📞 Contact & References

Refer to the included PDF documents for:
- **STGNN Intro.pdf**: Methodology and theoretical background
- **Graph Theory Project Analysis.pdf**: Detailed project analysis and results
