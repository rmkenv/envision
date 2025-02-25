import geopandas as gpd
import pandas as pd
import numpy as np
import folium
import requests
import json
from shapely.geometry import Point, LineString, Polygon, shape
from shapely.ops import transform
import pyproj
from functools import partial
import streamlit as st
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from arcgis.gis import GIS
from arcgis.features import FeatureLayer
from arcgis.raster import ImageryLayer
import matplotlib.colors as mcolors
import io
import base64
from PIL import Image

# Define buffer distances in miles
BUFFER_DISTANCES = {
    "1/10 mile": 0.1,
    "1/4 mile": 0.25,
    "1/2 mile": 0.5,
    "3/4 mile": 0.75,
    "1 mile": 1.0,
    "5 miles": 5.0
}

# Define scoring for each buffer
BUFFER_SCORES = {
    "1/10 mile": 6,
    "1/4 mile": 5,
    "1/2 mile": 4,
    "3/4 mile": 3,
    "1 mile": 2,
    "5 miles": 1
}

# Layer weights by category
LAYER_WEIGHTS = {
    "Justice 40": 1.0,
    "CDC SVI": 1.0,
    "Renewable": 0.9,
    "FEMA NRI": 1.1,
    "Flood Hazards": 1.2,
    "Biodiversity": 1.1,
    "Wildlife Refuge": 1.1,
    "Wetlands": 1.1,
    "Historic Places": 1.0,
    "Noise": 1.2,
    "EPA FRS": 1.2
}

# Layer metadata
LAYER_METADATA = {
    "Justice 40": {
        "url": "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/usa_november_2022/FeatureServer/0",
        "type": "polygon",
        "vintage": "2022",
        "scope": "CT",
        "query": "Assessment: Disadvantaged, Partially Disadvantaged, Not Disadvantaged"
    },
    "CDC SVI": {
        "url": "https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/CDC_SVI_2022_(Archive)/FeatureServer/2",
        "type": "polygon",
        "vintage": "2022",
        "scope": "CT",
        "query": "Sum of flags for the four themes"
    },
    "Renewable": {
        "url": "https://geoappext.nrcan.gc.ca/arcgis/rest/services/NACEI/energy_infrastructure_of_north_america_en/MapServer/28",
        "type": "point",
        "vintage": "2018",
        "scope": "Point"
    },
    "FEMA NRI": {
        "url": "https://services.arcgis.com/XG15cJAlne2vxtgt/arcgis/rest/services/National_Risk_Index_Census_Tracts/FeatureServer/0",
        "type": "polygon",
        "vintage": "2025",
        "scope": "CT",
        "query": "National Risk Rating Composite"
    },
    "Flood Hazards": {
        "url": "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/USA_Flood_Hazard_Reduced_Set_gdb/FeatureServer/0",
        "type": "line",
        "vintage": "2024",
        "scope": "Line",
        "query": "Not Zone X"
    },
    "Biodiversity": {
        "url": "https://tiledimageservices.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/AUBILayer/ImageServer",
        "type": "raster",
        "vintage": "2021",
        "scope": "Image"
    },
    "Wildlife Refuge": {
        "url": "https://services.arcgis.com/QVENGdaPbd4LUkLV/arcgis/rest/services/National_Wildlife_Refuge_System_Boundaries/FeatureServer/0",
        "type": "polygon",
        "vintage": "2025",
        "scope": "Image"
    },
    "Wetlands": {
        "url": "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/USA_Wetlands/FeatureServer/0",
        "type": "polygon",
        "vintage": "2024",
        "scope": "Poly"
    },
    "Historic Places": {
        "url": "https://mapservices.nps.gov/arcgis/services/cultural_resources/nrhp_locations/MapServer/WMSServer",
        "type": "point",
        "scope": "Point"
    },
    "Noise": {
        "url": "https://tiledimageservices.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/USA_Transportation_Noise___Rail_Road_and_Aviation_2020/ImageServer",
        "type": "raster",
        "vintage": "2020",
        "scope": "Image"
    },
    "EPA FRS": {
        "url": "https://services.arcgis.com/cJ9YHowT8TU7DUyn/arcgis/rest/services/FRS_INTERESTS/FeatureServer/0",
        "type": "point",
        "vintage": "2025",
        "scope": "Point",
        "query": "Hazardous waste sites, emissions, etc."
    }
}

# Function to create buffers around a parcel
def create_buffers(parcel_gdf):
    buffers = {}
    # Reproject to a projected CRS for accurate buffer distances
    parcel_proj = parcel_gdf.to_crs(epsg=3857)

    for distance_name, distance_miles in BUFFER_DISTANCES.items():
        # Convert miles to meters (1 mile = 1609.34 meters)
        distance_meters = distance_miles * 1609.34
        buffer = parcel_proj.copy()
        buffer['geometry'] = parcel_proj.geometry.buffer(distance_meters)
        # Convert back to WGS 84
        buffer = buffer.to_crs(epsg=4326)
        buffers[distance_name] = buffer

    return buffers

# Function to fetch and analyze features from a layer
def analyze_layer(layer_name, buffers, parcel_gdf):
    layer_info = LAYER_METADATA[layer_name]
    layer_url = layer_info["url"]
    layer_type = layer_info["type"]
    results = {}

    # Initialize GIS connection
    gis = GIS()

    try:
        if layer_type == "point" or layer_type == "line" or layer_type == "polygon":
            # Use ArcGIS API to access feature layers
            feature_layer = FeatureLayer(layer_url)

            for distance_name, buffer_gdf in buffers.items():
                # Get the buffer geometry as a dictionary for the query
                buffer_geom = buffer_gdf.geometry.iloc[0]
                geom_dict = json.loads(gpd.GeoSeries([buffer_geom]).__geo_interface__)['features'][0]['geometry']

                # Query features within the buffer
                query_result = feature_layer.query(
                    geometry=geom_dict,
                    geometry_type="esriGeometryPolygon",
                    spatial_relationship="esriSpatialRelIntersects",
                    return_count_only=True
                )

                count = query_result.get('count', 0)

                # Assign score based on count and buffer distance
                score = 0
                if count > 0:
                    score = BUFFER_SCORES[distance_name]
                    # For certain layers, we might adjust score based on count
                    if layer_name in ["EPA FRS", "Renewable", "Historic Places"]:
                        # Cap the count effect to avoid extreme scores
                        count_factor = min(count, 10) / 10
                        score = score * (1 + count_factor)

                results[distance_name] = {
                    "count": count,
                    "score": score
                }

        elif layer_type == "raster":
            # For raster layers, we'll use the ImageryLayer
            imagery_layer = ImageryLayer(layer_url)

            for distance_name, buffer_gdf in buffers.items():
                # Get statistics for the raster within the buffer
                buffer_geom = buffer_gdf.geometry.iloc[0]
                geom_dict = json.loads(gpd.GeoSeries([buffer_geom]).__geo_interface__)['features'][0]['geometry']

                # This is simplified - in practice you'd need to handle raster analysis properly
                # based on the specific raster data type
                try:
                    stats = imagery_layer.get_statistics(geometry=geom_dict)

                    # Determine if there's meaningful data in the buffer
                    has_data = stats.get('max', 0) > 0

                    score = BUFFER_SCORES[distance_name] if has_data else 0

                    results[distance_name] = {
                        "has_data": has_data,
                        "score": score
                    }
                except Exception as e:
                    results[distance_name] = {
                        "error": str(e),
                        "score": 0
                    }

    except Exception as e:
        # Handle errors for this layer
        for distance_name in BUFFER_DISTANCES.keys():
            results[distance_name] = {
                "error": str(e),
                "score": 0
            }

    return results

# Calculate overall scores
def calculate_scores(all_results):
    final_scores = {
        "by_distance": {},
        "by_layer": {},
        "total": 0,
        "weighted_total": 0
    }

    # Calculate scores by distance
    for distance_name in BUFFER_DISTANCES.keys():
        distance_score = 0
        for layer_name in LAYER_METADATA.keys():
            layer_result = all_results.get(layer_name, {}).get(distance_name, {})
            score = layer_result.get("score", 0)
            distance_score += score

        final_scores["by_distance"][distance_name] = distance_score

    # Calculate scores by layer
    for layer_name in LAYER_METADATA.keys():
        layer_score = 0
        layer_weight = LAYER_WEIGHTS.get(layer_name, 1.0)

        for distance_name in BUFFER_DISTANCES.keys():
            layer_result = all_results.get(layer_name, {}).get(distance_name, {})
            score = layer_result.get("score", 0)
            layer_score += score

        weighted_layer_score = layer_score * layer_weight
        final_scores["by_layer"][layer_name] = {
            "raw_score": layer_score,
            "weight": layer_weight,
            "weighted_score": weighted_layer_score
        }

        final_scores["total"] += layer_score
        final_scores["weighted_total"] += weighted_layer_score

    return final_scores

# Create visualization map
def create_map(parcel_gdf, buffers, all_results):
    # Create a folium map centered on the parcel
    center = [parcel_gdf.geometry.iloc[0].centroid.y, parcel_gdf.geometry.iloc[0].centroid.x]
    mymap = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    # Add parcel
    folium.GeoJson(
        parcel_gdf.__geo_interface__,
        name="Parcel",
        style_function=lambda x: {"color": "red", "fillColor": "red", "weight": 2, "fillOpacity": 0.5}
    ).add_to(mymap)

    # Add buffers with different colors
    colors = ["#ffffcc", "#d9f0a3", "#addd8e", "#78c679", "#41ab5d", "#238443", "#005a32"]
    for i, (distance_name, buffer_gdf) in enumerate(buffers.items()):
        color = colors[i % len(colors)]
        folium.GeoJson(
            buffer_gdf.__geo_interface__,
            name=f"Buffer {distance_name}",
            style_function=lambda x, color=color: {"color": color, "fillColor": color, "weight": 1, "fillOpacity": 0.1}
        ).add_to(mymap)

    # Add layer control
    folium.LayerControl().add_to(mymap)

    return mymap

# Streamlit dashboard
def create_dashboard():
    st.title("Parcel Environmental and Social Impact Assessment")

    st.header("Upload Parcel GeoJSON")
    uploaded_file = st.file_uploader("Choose a GeoJSON file", type="geojson")

    if uploaded_file is not None:
        # Load the parcel data
        parcel_data = json.load(uploaded_file)
        parcel_gdf = gpd.GeoDataFrame.from_features(
            parcel_data["features"] if "features" in parcel_data else [parcel_data],
            crs="EPSG:4326"
        )

        # Create buffers
        buffers = create_buffers(parcel_gdf)

        # Run analysis (this could take time depending on the layers)
        st.header("Analyzing Proximity to Environmental and Social Factors")
        progress_bar = st.progress(0)

        all_results = {}
        for i, layer_name in enumerate(LAYER_METADATA.keys()):
            st.write(f"Analyzing {layer_name}...")
            all_results[layer_name] = analyze_layer(layer_name, buffers, parcel_gdf)
            progress_bar.progress((i + 1) / len(LAYER_METADATA))

        # Calculate scores
        scores = calculate_scores(all_results)

        # Display map
        st.header("Parcel and Buffer Zones")
        mymap = create_map(parcel_gdf, buffers, all_results)
        folium_static(mymap)

        # Display scores
        st.header("Assessment Scores")

        # Weighted total score
        st.subheader("Overall Impact Score")
        total_score = scores["weighted_total"]
        max_possible = sum([6 * len(BUFFER_DISTANCES) * weight for weight in LAYER_WEIGHTS.values()])
        score_percentage = (total_score / max_possible) * 100

        # Create a gauge for the overall score
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score_percentage,
            title={"text": "Overall Impact Score (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 33], "color": "green"},
                    {"range": [33, 66], "color": "yellow"},
                    {"range": [66, 100], "color": "red"}
                ]
            }
        ))
        st.plotly_chart(fig)

        # Scores by layer
        st.subheader("Scores by Layer")
        layer_scores = []
        for layer_name, score_data in scores["by_layer"].items():
            layer_scores.append({
                "Layer": layer_name,
                "Raw Score": score_data["raw_score"],
                "Weight": score_data["weight"],
                "Weighted Score": score_data["weighted_score"]
            })

        layer_scores_df = pd.DataFrame(layer_scores)
        st.dataframe(layer_scores_df.sort_values("Weighted Score", ascending=False))

        # Bar chart of layer scores
        fig = px.bar(
            layer_scores_df.sort_values("Weighted Score", ascending=False),
            x="Layer",
            y="Weighted Score",
            color="Weighted Score",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(title="Impact Scores by Layer")
        st.plotly_chart(fig)

        # Scores by distance
        st.subheader("Scores by Buffer Distance")
        distance_scores = []
        for distance_name, score in scores["by_distance"].items():
            distance_scores.append({
                "Distance": distance_name,
                "Score": score
            })

        distance_scores_df = pd.DataFrame(distance_scores)

        # Order by distance
        distance_order = list(BUFFER_DISTANCES.keys())
        distance_scores_df["Distance"] = pd.Categorical(
            distance_scores_df["Distance"],
            categories=distance_order,
            ordered=True
        )
        distance_scores_df = distance_scores_df.sort_values("Distance")

        # Line chart of scores by distance
        fig = px.line(
            distance_scores_df,
            x="Distance",
            y="Score",
            markers=True,
            line_shape="spline"
        )
        fig.update_layout(title="Cumulative Impact by Distance")
        st.plotly_chart(fig)

        # Detailed results for each layer
        st.header("Detailed Analysis")
        for layer_name, layer_results in all_results.items():
            with st.expander(f"{layer_name} Details"):
                st.write(f"**Layer Type:** {LAYER_METADATA[layer_name]['type']}")
                st.write(f"**Vintage:** {LAYER_METADATA[layer_name].get('vintage', 'N/A')}")

                # Create table of results
                details = []
                for distance_name, result in layer_results.items():
                    details.append({
                        "Buffer Distance": distance_name,
                        "Count/Value": result.get("count", "N/A"),
                        "Score": result.get("score", 0)
                    })

                details_df = pd.DataFrame(details)
                st.dataframe(details_df)

if __name__ == "__main__":
    create_dashboard()
