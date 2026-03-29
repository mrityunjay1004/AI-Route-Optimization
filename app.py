import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from geopy.geocoders import Nominatim
import joblib
import time
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2

st.set_page_config(page_title="AI Route Optimization System", layout="wide")
st.title("🚚 AI Route Optimization System")

# -----------------------------
# GEOCODER
# -----------------------------
geolocator = Nominatim(user_agent="route_optimizer")

@st.cache_data
def get_coordinates(city):
    try:
        location = geolocator.geocode(city + ", India", timeout=10)
        if location:
            return location.latitude, location.longitude
        return None, None
    except:
        return None, None

# -----------------------------
# LOAD MODEL
# -----------------------------
try:
    model = joblib.load("eta_model.pkl")
except:
    model = None
    st.warning("ETA model not found.")

# -----------------------------
# SESSION STATE
# -----------------------------
if "locations" not in st.session_state:
    st.session_state.locations = []

# -----------------------------
# CSV UPLOAD
# -----------------------------
st.subheader("Upload Dataset")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df_csv = pd.read_csv(uploaded_file)
    st.session_state.locations = df_csv.to_dict("records")
    st.success("Dataset uploaded successfully!")

# -----------------------------
# INPUT FORM
# -----------------------------
st.subheader("Add Delivery Order")

with st.form("order_form"):

    col1, col2 = st.columns(2)

    with col1:
        order_id = st.number_input("Order ID", step=1)
        pickup_city = st.text_input("Pickup City")
        package_weight = st.number_input("Package Weight (kg)", min_value=0.1)
        vehicle_type = st.selectbox("Vehicle Type", ["Bike", "Van", "Truck"])

    with col2:
        delivery_city = st.text_input("Delivery City")
        traffic_level = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
        delivery_priority = st.selectbox("Delivery Priority", ["Low", "Medium", "High"])

    col3, col4 = st.columns(2)

    with col3:
        delivery_cost = st.number_input("Delivery Cost (₹)", min_value=0.0)

    with col4:
        delivery_revenue = st.number_input("Delivery Revenue (₹)", min_value=0.0)

    submit = st.form_submit_button("Add Order")

    if submit:
        if pickup_city and delivery_city:
            profit = delivery_revenue - delivery_cost

            st.session_state.locations.append({
                "order_id": order_id,
                "pickup_city": pickup_city,
                "delivery_city": delivery_city,
                "package_weight_kg": package_weight,
                "vehicle_type": vehicle_type,
                "traffic_level": traffic_level,
                "delivery_priority": delivery_priority,
                "delivery_cost": delivery_cost,
                "delivery_revenue": delivery_revenue,
                "profit_per_delivery": profit
            })

            st.success("Order added successfully!")
        else:
            st.error("Enter both cities")

# -----------------------------
# MAIN LOGIC
# -----------------------------
if len(st.session_state.locations) > 0:

    df = pd.DataFrame(st.session_state.locations)
    st.subheader("Current Orders")
    st.dataframe(df)

    # -----------------------------
    # GEOCODING
    # -----------------------------
    city_coords = {}
    cities = list(set(df["pickup_city"]).union(set(df["delivery_city"])))

    progress = st.progress(0)

    for i, city in enumerate(cities):
        lat, lon = get_coordinates(city)
        city_coords[city] = (lat, lon)
        time.sleep(1)
        progress.progress((i + 1) / len(cities))

    valid_cities = [c for c in cities if city_coords[c][0] is not None]

    if len(valid_cities) < 2:
        st.error("Need at least 2 valid cities")
        st.stop()

    coords = [city_coords[c] for c in valid_cities]

    # -----------------------------
    # HAVERSINE
    # -----------------------------
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
        c = 2*atan2(sqrt(a), sqrt(1-a))
        return R*c

    # -----------------------------
    # DISTANCE MATRIX
    # -----------------------------
    size = len(coords)
    matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if i != j:
                matrix[i][j] = haversine(
                    coords[i][0], coords[i][1],
                    coords[j][0], coords[j][1]
                )

    # -----------------------------
    # ROUTE OPTIMIZATION
    # -----------------------------
    manager = pywrapcp.RoutingIndexManager(len(matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return int(matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)] * 100)

    transit_callback = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)

    route = []
    index = routing.Start(0)

    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))

    route.append(manager.IndexToNode(index))

    route_cities = [valid_cities[i] for i in route if i < len(valid_cities)]

    st.subheader("Optimized Route")
    st.success(" ➝ ".join(route_cities))

    # -----------------------------
    # MAP
    # -----------------------------
    m = folium.Map(location=coords[0], zoom_start=5)

    for city, (lat, lon) in city_coords.items():
        if lat:
            folium.Marker([lat, lon], popup=city).add_to(m)

    path = [city_coords[c] for c in route_cities if city_coords[c][0]]
    folium.PolyLine(path, color="blue", weight=4).add_to(m)

    st_folium(m, width=900)

    # -----------------------------
    # INSIGHTS
    # -----------------------------
    st.subheader("📊 Route Optimization Insights")

    fuel_efficiency = 15
    fuel_price = 100
    avg_speed = 40

    naive_distance = sum(
        haversine(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])
        for i in range(len(coords) - 1)
    )

    optimized_distance = sum(
        haversine(coords[route[i]][0], coords[route[i]][1],
                  coords[route[i+1]][0], coords[route[i+1]][1])
        for i in range(len(route) - 1)
    )

    fuel_saved = (naive_distance - optimized_distance) / fuel_efficiency
    time_saved = (naive_distance - optimized_distance) / avg_speed
    profit_gain = fuel_saved * fuel_price

    # Percentage savings
    distance_saving_pct = ((naive_distance - optimized_distance) / naive_distance) * 100
    fuel_saving_pct = distance_saving_pct
    time_saving_pct = distance_saving_pct

    col1, col2, col3 = st.columns(3)

    col1.metric("⛽ Fuel Saved (L)", round(fuel_saved, 2), f"{round(fuel_saving_pct,1)}%")
    col2.metric("⏱ Time Saved (hrs)", round(time_saved, 2), f"{round(time_saving_pct,1)}%")
    col3.metric("💰 Profit Increase (₹)", round(profit_gain, 2))

    # -----------------------------
    # GRAPH
    # -----------------------------
    st.subheader("📊 Before vs After Comparison")

    metrics = ["Distance", "Fuel", "Time"]

    naive_values = [
        naive_distance,
        naive_distance / fuel_efficiency,
        naive_distance / avg_speed
    ]

    optimized_values = [
        optimized_distance,
        optimized_distance / fuel_efficiency,
        optimized_distance / avg_speed
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots()

    ax.bar(x - width/2, naive_values, width, label="Before")
    ax.bar(x + width/2, optimized_values, width, label="After")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("Optimization Impact")
    ax.legend()

    st.pyplot(fig)

else:
    st.info("Upload dataset or add delivery orders to start optimization.")
