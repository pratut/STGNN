import numpy as np 
import pandas as pd 
import torch 
import os 
from tqdm import tqdm 
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric_temporal.nn.attention.stgcn import STConv

## Geodesic Distance
from geopy.distance import geodesic
from geopy.point import Point

def geodesic_distance(lat1, lon1, lat2, lon2):
    # Create Point objects for the coordinates
    point1 = Point(latitude=lat1, longitude=lon1)
    point2 = Point(latitude=lat2, longitude=lon2)

    # Calculate the geodesic distance using Vincenty formula
    distance = geodesic(point1, point2).kilometers

    return distance

## DataLoader
class WeatherDatasetLoader(object):

    def __init__(self, snapshots, edge_index, edge_weight):
        self._snapshots = snapshots
        self._edge_index = edge_index
        self._edge_weight = edge_weight

    def _get_edge_index(self):
        self._edges = self._edge_index

    def _get_edge_weights(self):
        self._edge_weights = self._edge_weight

    def _get_targets_and_features(self):
        stacked_target = self._snapshots

        # self.features = [np.expand_dims(stacked_target[0: self.lags, :, :], axis=0),]
        self.features = [np.expand_dims(stacked_target[:self.lags], axis=0),]

        self.targets = [np.expand_dims(np.transpose(stacked_target[-self._pred_seq:, :, [0, 1, 2, 4, 5, 7, 8]].T, (2, 1, 0)), axis=0),]
        #self.targets = [
        #             np.expand_dims
        #             (np.transpose(stacked_target[-self._pred_seq:, :, [0, 1, 2, 4, 5, 7, 8]].T, (2, 1, 0)),
        #              axis=0)
        #             for i in range(stacked_target.shape[0] - self.lags - self._pred_days)
        #         ]

    def get_dataset(self, lags: int = 60, pred_days=24) -> StaticGraphTemporalSignal:
        self.lags = lags
        self._pred_seq = pred_days
        self._get_edge_index()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features , self.targets
        )
        return dataset

# ## DataLoader
# class WeatherDatasetLoader(object):
#
#     def __init__(self, snapshots, edge_index, edge_weight):
#         self._snapshots = snapshots
#         self._edge_index = torch.tensor(edge_index)
#         self._edge_weight = torch.tensor(edge_weight)
#
#     def _get_edge_index(self):
#         self._edges = self._edge_index
#
#     def _get_edge_weights(self):
#         self._edge_weights = self._edge_weight
#
#     def _get_targets_and_features(self):
#         stacked_target = self._snapshots
#         self.features = [
#             np.expand_dims(stacked_target[i: i + self.lags, :, :], axis=0)
#             for i in range(stacked_target.shape[0] - self.lags - self._pred_days)
#         ]
#
#         self.targets = [
#             np.expand_dims
#             (np.transpose(stacked_target[i + self.lags:(i + self.lags + self._pred_days), :, [0, 1, 2, 4, 5, 7, 8]].T, (2, 1, 0)),
#              axis=0)
#             for i in range(stacked_target.shape[0] - self.lags - self._pred_days)
#         ]
#
#     def get_dataset(self, lags: int = 60, pred_days = 24) -> StaticGraphTemporalSignal:
#         self.lags = lags
#         self._pred_days = pred_days
#         self._get_edge_index()
#         self._get_edge_weights()
#         self._get_targets_and_features()
#         dataset = StaticGraphTemporalSignal(
#             self._edges, self._edge_weights, self.features, self.targets
#         )
#         return dataset

"""
    def _get_targets_and_features(self):
        stacked_target = self._snapshots
        self.features = [
            np.expand_dims(stacked_target[i: i + self.lags, :, :], axis=0)
            for i in range(stacked_target.shape[0] - self.lags - self._pred_days)
        ]

        self.targets = [
            np.expand_dims
            (np.transpose(stacked_target[i + self.lags:(i + self.lags + self._pred_days), :, [0, 1, 2, 4, 5, 7, 8]].T, (2, 1, 0)),
             axis=0)
            for i in range(stacked_target.shape[0] - self.lags - self._pred_days)
        ]

    def get_dataset(self, lags: int = 60, pred_days = 24) -> StaticGraphTemporalSignal:
        self.lags = lags
        self._pred_days = pred_days
        self._get_edge_index()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset
"""

def normalizeTestData(test_file_path, mean_file_path, std_file_path):
    df = pd.read_csv(test_file_path)
    selected_features = ['T2M', 'T2MWET', 'TS', 'T2M_RANGE', 'T2M_MAX', 'T2M_MIN', 'QV2M', 'RH2M', 'PRECTOTCORR', 'PS',
                            'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX',
                            'WS50M_MIN', 'WS50M_RANGE']
    data = df[selected_features]
    
    mean_values = pd.read_csv(mean_file_path)
    std_values = pd.read_csv(std_file_path)

    for index, row in data.iterrows():
        # Normalize the row using the corresponding mean and standard deviation values
        normalized_row = (row - mean_values.iloc[0]) / std_values.iloc[0]
        # Update the row in df with the normalized values
        data.loc[index] = normalized_row

    # normalized_data = data #(data - mean_values) / std_values
    df[selected_features] = data[selected_features]
    return df, mean_values, std_values


def get_features(df, stations):   
    target_features = ['T2M', 'T2MWET', 'TS', 'T2M_RANGE', 'T2M_MAX', 'T2M_MIN', 'QV2M', 'RH2M', 'PRECTOTCORR', 'PS',
                            'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX',
                            'WS50M_MIN', 'WS50M_RANGE']
    #                    0        1          2           3       4       5     6      7        8
    # Our target labels: [4, 5, 6] -> ['T2M_MIN', 'RH2M', 'PRECTOTCORR']
    
    STATIONS_SNAPSHOTS = []
    
    # the `pd.Categorical` function is used to convert the 'Location' column to a categorical type with a 
    # custom order specified by the `custom_order` list. The `ordered=True` argument ensures that the custom 
    # order is respected when performing operations like `groupby`.
    df['Location'] = pd.Categorical(df['Location'], categories=stations, ordered=True)

    grouped_df = df.groupby('Location')
    
    for _, group in tqdm(grouped_df):
        # Append the features for each station to the list
        snapshot = group[target_features].values.tolist()
        STATIONS_SNAPSHOTS.append(snapshot)
    
    return STATIONS_SNAPSHOTS

def get_stations(filename):
    with open(filename, 'r') as f:
        stations = [line.rstrip('\n') for line in f]
    return stations


## Model for lags=43, pred_seq = 7 
class STGCN_Best_Babu(torch.nn.Module):
    """
    Processes a sequence of graph data to produce a spatio-temporal embedding
    to be used for regression, classification, clustering, etc.
    Time_seq predict (days) = Lags -2(Kernal size -1)*No. of STCOnv = 31-2(7-1) =8
    """

    def __init__(self):
        super(STGCN_Best_Babu, self).__init__()
        self.stconv_block1 = STConv(750, 18, 32, 64, 9, 3)
        self.stconv_block2 = STConv(750, 64, 128, 256, 7, 3)
        self.stconv_block3 = STConv(750, 256, 128, 64, 5, 3)
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 7)

    def forward(self, x, edge_index, edge_attr):
        temp = self.stconv_block1(x, edge_index, edge_attr)
        temp = self.stconv_block2(temp, edge_index, edge_attr)
        temp = self.stconv_block3(temp, edge_index, edge_attr)
        temp = self.fc1(temp)
        temp = self.fc2(temp)

        return temp




if __name__ == "__main__":
    
    lags = 60
    pred_days = 24
    
    test_data_path = 'D:/Local Government/Test_Data/test_data_updated_750.csv'
    stations_path = 'D:/Local Government/Test_Code/stations.txt'
    locations_path = "D:/Local Government/Test_Data/municipality_geometry_750_updated.csv"
    weights_path = 'D:/Local Government/Test_Code/Model_60Lags_STConv_Best_March5.pt'
    edge_index_path = 'D:/Local Government/Test_Code/edge_index.pt'
    edge_weight_path = 'D:/Local Government/Test_Code/edge_weights.pt'
    
    mean_file_path = "D:/Local Government/Test_Code/mean_values.csv"
    std_file_path = "D:/Local Government/Test_Code/std_values.csv"
    
    
    
    stations = get_stations(stations_path)
    df, mean_values, std_values = normalizeTestData(test_data_path, mean_file_path, std_file_path)

    snapshot = get_features(df, stations)
    snapshot = np.array(snapshot)
    snap_transpose = np.transpose(snapshot, (1, 0, 2))

    lags_ = snap_transpose.shape[0]

    if lags_ < lags:
        error_message = (
            f"Error: Number of lags in test data ({lags_}) is less than "
            f"the number of lags in the input sequence ({lags}). "
            "Please make sure that the test data has enough lags to "
            "cover the input sequence lags. Terminating the program."
        )
        raise ValueError(error_message)


    # print(f"snapshots: {snap_transpose.shape}")

    edge_index = torch.load(edge_index_path)#.to(torch.float32)
    edge_weight = torch.load(edge_weight_path).to(torch.float32)

    loader = WeatherDatasetLoader(snapshots=snap_transpose, 
                                    edge_index=edge_index,
                                    edge_weight=edge_weight)
    test_dataset = loader.get_dataset(lags=lags, pred_days=pred_days)

    torch.cuda.empty_cache()

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the selected device
    model = STGCN_Best_Babu().to(device)

    # Load the model on CPU if CUDA is not available
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

    #####################
    ## Evaluation mode on
    #####################
    model.eval()

    # Load the data on CPU if CUDA is not available
    for data in test_dataset:
        snapshot = data

    # Move the data to the selected device
    snapshot = snapshot.to(device)
    y_true = snapshot.y
    y_pred = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    #print(y_pred)

    ################
    ## de-normalize 
    ################
    # target_feat = ['T2M_MIN', 'RH2M', 'PRECTOTCORR']
    mean_tensor = torch.tensor(mean_values.iloc[:,[0, 1, 2, 4, 5, 7, 8]].values, dtype=torch.float32)
    std_tensor = torch.tensor(std_values.iloc[:,[0, 1, 2, 4, 5, 7, 8]].values, dtype=torch.float32)

    y_pred_ = torch.squeeze(y_pred)
    y_true_ = torch.squeeze(y_true)
    mean_tensor_broadcasted = np.expand_dims(mean_tensor.detach().numpy(), axis=0)
    std_tensor_broadcasted = np.expand_dims(std_tensor.detach().numpy(), axis=0)

    y_pred_ = y_pred_.cpu().detach().numpy()
    y_true_ = y_true_.cpu().detach().numpy()

    # De-normalize y_pred_
    y_pred_denormalized = (y_pred_ * std_tensor_broadcasted) + mean_tensor_broadcasted
    y_true_denormalized = (y_true_ * std_tensor_broadcasted) + mean_tensor_broadcasted

    mask = y_pred_denormalized[:, :, -1]<0
    y_pred_denormalized[mask, -1] = 0

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y_pred_denormalized.flatten(), y_true_denormalized.flatten())
    mae = mean_absolute_error(y_pred_denormalized.flatten(), y_true_denormalized.flatten())
    r_squared = r2_score(y_pred_denormalized.flatten(), y_true_denormalized.flatten())
    rmse = np.sqrt(mse)
    print("Mean Square Error:", mse)
    print("Mean Absolute Error:", mae)
    print("Root Mean Square Error:", rmse)
    print("R-Squared:", r_squared)

    import matplotlib.pyplot as plt
    target = np.transpose(y_true_denormalized, (1, 2, 0))
    predicted = np.transpose(y_pred_denormalized,(1, 2, 0))

    plt.plot(target[1][0])
    plt.plot(predicted[1][0])
    plt.show()

    rmse_total = []
    for idx in range(750):
        mse = mean_squared_error(target[idx].flatten(), predicted[idx].flatten())
        rmse = np.sqrt(mse)
        rmse_total.append(rmse)

    plt.plot(rmse_total)
    plt.show()




    # #### Let's say the new location captured by the app is as follows.
    # # lat_ = [lat, long]
    # lat_ = [28.6616, 80.6392]
    #
    # df_locations = pd.read_csv(locations_path)
    # df_locations['Distance'] = df_locations.apply(lambda row: geodesic_distance(*lat_, row['latitude'], row['longitude']), axis=1)
    #
    # # Find the index of the row with the smallest distance
    # min_distance_index = df_locations['Distance'].idxmin()
    # # print("Index of the row with the smallest distance: ", min_distance_index)
    #
    # min_distance_index2 = df_locations.index.get_loc(min_distance_index)
    # # print("Index of the row with the smallest distance: ", min_distance_index2)
    #
    # y_pred_new_location = y_pred_denormalized[:, min_distance_index2, :]
    # #print(y_pred_new_location)
    # # Set print options
    # np.set_printoptions(suppress=True, precision=4)
    # print(f"y_pred for new locations (for ({pred_days}) days): \n\n{y_pred_new_location}")
    