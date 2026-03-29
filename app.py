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
from math import radians, sin, cos, sqrt, atan2

st.set_page_config(page_title="AI Route Optimization System", layout="wide")

st.title("🚚 AI Route Optimization System")

# -----------------------------
# GEOCODER (FIX 1: CACHING)
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
# LOAD ETA MODEL
# -----------------------------

try:
    model = joblib.load("eta_model.pkl")
except:
    st.warning("ETA model not found. Please run train_model.py first.")
    model = None

# -----------------------------
# SESSION STATE
# -----------------------------

if "locations" not in st.session_state:
    st.session_state.locations = []

# -----------------------------
# CSV UPLOAD
# -----------------------------

st.subheader("Upload Logistics Dataset")

uploaded_file = st.file_uploader(
    "Upload indian_logistics_city_dataset.csv",
    type=["csv"]
)

if uploaded_file is not None:
    df_csv = pd.read_csv(uploaded_file)
    st.session_state.locations = df_csv.to_dict("records")
    st.success("Dataset uploaded successfully!")

# -----------------------------
# MANUAL ORDER INPUT
# -----------------------------

st.subheader("Add Full Delivery Order")

with st.form("order_form"):

    col1, col2 = st.columns(2)

    with col1:
        order_id = st.number_input("Order ID", step=1)
        pickup_city = st.text_input("Pickup City")
        package_weight = st.number_input("Package Weight (kg)", min_value=0.1)

        vehicle_type = st.selectbox(
            "Vehicle Type",
            ["Bike", "Van", "Truck"]
        )

    with col2:
        delivery_city = st.text_input("Delivery City")

        traffic_level = st.selectbox(
            "Traffic Level",
            ["Low", "Medium", "High"]
        )

        delivery_priority = st.selectbox(
            "Delivery Priority",
            ["Low", "Medium", "High"]
        )

    st.markdown("### Optional Business Inputs")

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

            st.success("✅ Full order added successfully!")

        else:
            st.error("❌ Please enter both pickup and delivery cities.")

# -----------------------------
# PROCESS DATA
# -----------------------------

if len(st.session_state.locations) > 0:

    df = pd.DataFrame(st.session_state.locations)

    st.subheader("Current Orders")
    st.dataframe(df)

    # -----------------------------
    # GET COORDINATES
    # -----------------------------

    st.subheader("Fetching Coordinates (Geocoding...)")

    city_coords = {}

    cities = list(set(df["pickup_city"]).union(set(df["delivery_city"])))

    progress = st.progress(0)

    for i, city in enumerate(cities):

        lat, lon = get_coordinates(city)
        city_coords[city] = (lat, lon)

        time.sleep(1)  # FIX 2: delay to avoid API blocking

        progress.progress((i + 1) / len(cities))

    # -----------------------------
    # FILTER VALID CITIES
    # -----------------------------

    valid_cities = []

    for c in cities:
        if city_coords[c][0] is not None:
            valid_cities.append(c)

    # FIX 3: prevent crash
    if len(valid_cities) < 2:
        st.error("❌ Need at least 2 valid cities for route optimization.")
        st.stop()

    coords = [city_coords[c] for c in valid_cities]

    # -----------------------------
    # HAVERSINE FUNCTION
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

    try:

        manager = pywrapcp.RoutingIndexManager(len(matrix), 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):

            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)

            return int(matrix[from_node][to_node] * 100)

        transit_callback = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        solution = routing.SolveWithParameters(search_parameters)

        if solution is None:
            st.error("❌ Could not find an optimized route.")
            st.stop()

        route = []

        index = routing.Start(0)

        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))

        route.append(manager.IndexToNode(index))

    except Exception as e:
        st.error(f"Route optimization failed: {e}")
        st.stop()

    # -----------------------------
    # SHOW ROUTE
    # -----------------------------

    st.subheader("Optimized Route")

    route_cities = [valid_cities[i] for i in route if i < len(valid_cities)]

    st.success(" ➝ ".join(route_cities))

    # -----------------------------
    # MAP
    # -----------------------------

    st.subheader("Route Map")

    m = folium.Map(location=coords[0], zoom_start=5)

    for city, (lat, lon) in city_coords.items():
        if lat is not None:
            folium.Marker([lat, lon], popup=city).add_to(m)

    path = [city_coords[c] for c in route_cities if city_coords[c][0] is not None]

    folium.PolyLine(path, color="blue", weight=4).add_to(m)

    st_folium(m, width=900)

else:
    st.info("Upload dataset or add delivery orders to start optimization.")
    # -----------------------------
# INSIGHTS CALCULATION
# -----------------------------

st.subheader("📊 Route Optimization Insights")

# Fuel efficiency assumption
fuel_efficiency = 15  # km per liter
fuel_price = 100  # ₹ per liter

# Average speed assumption
avg_speed = 40  # km/h

# -----------------------------
# NAIVE ROUTE DISTANCE
# -----------------------------

naive_distance = 0

for i in range(len(coords) - 1):
    naive_distance += haversine(
        coords[i][0], coords[i][1],
        coords[i+1][0], coords[i+1][1]
    )

# -----------------------------
# OPTIMIZED ROUTE DISTANCE
# -----------------------------

optimized_distance = 0

for i in range(len(route) - 1):
    a = coords[route[i]]
    b = coords[route[i+1]]

    optimized_distance += haversine(a[0], a[1], b[0], b[1])

# -----------------------------
# FUEL CALCULATION
# -----------------------------

fuel_naive = naive_distance / fuel_efficiency
fuel_optimized = optimized_distance / fuel_efficiency

fuel_saved = fuel_naive - fuel_optimized

# -----------------------------
# TIME CALCULATION
# -----------------------------

time_naive = naive_distance / avg_speed
time_optimized = optimized_distance / avg_speed

time_saved = time_naive - time_optimized

# -----------------------------
# PROFIT CALCULATION
# -----------------------------

if "delivery_revenue" in df.columns and "delivery_cost" in df.columns:

    total_revenue = df["delivery_revenue"].sum()

    cost_naive = fuel_naive * fuel_price
    cost_optimized = fuel_optimized * fuel_price

    profit_naive = total_revenue - cost_naive
    profit_optimized = total_revenue - cost_optimized

    profit_gain = profit_optimized - profit_naive

else:
    profit_naive = profit_optimized = profit_gain = 0

# -----------------------------
# DISPLAY METRICS
# -----------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "⛽ Fuel Saved (Liters)",
    round(fuel_saved, 2)
)

col2.metric(
    "⏱ Time Saved (Hours)",
    round(time_saved, 2)
)

col3.metric(
    "💰 Profit Increase (₹)",
    round(profit_gain, 2)
)

# -----------------------------
# EXTRA INSIGHTS
# -----------------------------

st.markdown("### Detailed Comparison")

comparison_df = pd.DataFrame({
    "Metric": ["Distance (km)", "Fuel Used (L)", "Time (hrs)", "Profit (₹)"],
    "Naive Route": [
        round(naive_distance,2),
        round(fuel_naive,2),
        round(time_naive,2),
        round(profit_naive,2)
    ],
    "Optimized Route": [
        round(optimized_distance,2),
        round(fuel_optimized,2),
        round(time_optimized,2),
        round(profit_optimized,2)
    ]
})

st.dataframe(comparison_df)
