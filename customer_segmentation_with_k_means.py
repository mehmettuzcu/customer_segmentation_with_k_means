########################## Libraries and Utilities ##########################

import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings
warnings.simplefilter(action='ignore', category=Warning)

from helpers.eda import *
from helpers.data_prep import *

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


#####################  Exploratory Data Analysis ###########################

data = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = data.copy()

df.shape
check_df(df)

# Get number of Null values in a dataframe
df.isnull().sum()

# Remove missing observations from the data set
df.dropna(inplace=True)

# Removing canceled transactions from the dataset
df = df[~df["Invoice"].str.contains("C", na=False)]

# Total earnings per invoice
df["TotalPrice"] = df["Quantity"] * df["Price"]
df.head()

###### Calculating RFM Metrics #####

# Recency: Indicates how many days ago the customer arrived.
# Frequency: It is frequency. Indicates how many purchases the customer has made.
# Monetary: It is the total monetary value made by the customer.

df["InvoiceDate"].max()  # last date of registration in shopping
today_date = dt.datetime(2011, 12, 11)  # from the last date to the date of analysis  (+2)

rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                     'Invoice': lambda num: num.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

rfm.head()

rfm.columns = ['recency', 'frequency', 'monetary']
rfm.head()

rfm = rfm[rfm["monetary"] > 0]

rfm.shape

rfm.describe([0.05, 0.01, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T

################################# K-Means #################################

# Standardization process so that distance-based methods are not affected by outliers
sc = MinMaxScaler((0, 1))
std_rfm = sc.fit_transform(rfm)

std_rfm[0:5]

kmeans = KMeans(n_clusters=5)
k_fit = kmeans.fit(std_rfm)


k_fit.cluster_centers_
k_fit.labels_
k_fit.inertia_


################################# Visualization of Clusters ################################

k_means = KMeans(n_clusters=2).fit(std_rfm)
kumeler = k_means.labels_
type(df)
std_rfm = pd.DataFrame(std_rfm)

plt.scatter(std_rfm.iloc[:, 0],
            std_rfm.iloc[:, 1],
            c=kumeler,
            s=50,
            cmap="viridis")
plt.show()
ü
merkezler = k_means.cluster_centers_

plt.scatter(std_rfm.iloc[:, 0],
            std_rfm.iloc[:, 1],
            c=kumeler,
            s=50,
            cmap="viridis")

plt.scatter(merkezler[:, 0],
            merkezler[:, 1],
            c="red",
            s=200,
            alpha=0.8)
plt.show()

################################# Determining the Optimum Number of Clusters ################################

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(std_rfm)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık Uzaklık Artık Toplamları")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

# A more automated way:
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 50))  # After doing iteration 30, the inertia value is 12.75 and the class is 7. I achieved 8 classes and 10.87 inertia in 50 iterations.
elbow.fit(std_rfm)
elbow.show()

elbow.elbow_value_

################################# Final Clusters ################################

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(std_rfm)
kumeler = kmeans.labels_

pd.DataFrame({"Customer ID": rfm.index, "Kumeler": kumeler})

rfm["cluster_no"] = kumeler

rfm["cluster_no"] = rfm["cluster_no"] + 1

rfm.head()

rfm.groupby("cluster_no").agg({"cluster_no": "count"})
rfm.groupby("cluster_no").agg(np.mean)

rfm[rfm["cluster_no"] == 5]

rfm[rfm["cluster_no"] == 6]

rfm.columns

def box_plot(data, cluster_no, col):
    sns.boxplot(x="cluster_no", y=col, data=data)
    plt.show()

for col in rfm.columns:
    box_plot(rfm, "cluster_no", col)

df["TotalPrice"] = df["Quantity"] * df["Price"]


df = data.copy()

###################### Functionalization of all steps ######################

def create_rfm(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]

    df["InvoiceDate"].max()
    today_date = dt.datetime(2011, 12, 11)
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    rfm.columns = ['recency', 'frequency', 'monetary']
    rfm = rfm[rfm["monetary"] > 0]


    sc = MinMaxScaler((0, 1))
    rfm_std = sc.fit_transform(rfm)

    kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(rfm_std)
    print(f"Clusters: {kmeans.n_clusters}")
    # print(f"Labels: {kmeans.labels_}")
    print(f"Inertia: {kmeans.inertia_}")
    kumeler = kmeans.labels_

    pd.DataFrame({"Customer ID": rfm.index, "Kumeler": kumeler})

    rfm["cluster_no"] = kumeler

    rfm["cluster_no"] = rfm["cluster_no"] + 1

    return rfm

rfm = create_rfm(df)


rfm.groupby("cluster_no").agg({"cluster_no": "count"})
rfm.groupby("cluster_no").agg(np.mean)
