import folium
from folium.plugins import HeatMap, MarkerCluster

def get_priority_color(priority):
    if priority == 'Very High':
        return 'red'
    elif priority == 'High':
        return 'orange'
    elif priority == 'Medium':
        return 'gold'
    else:
        return 'blue'

def build_investigation_map(
    last_known_coords,
    frequent_areas=None,
    predicted_rankings=None,
    probable_route_coords=None,
    anomalous_points=None,
    raw_trajectories_df=None,
    zoom_start=12
):
    """
    Generate an interactive Folium map for missing person location investigation.
    """
    center_lat, center_lon = last_known_coords
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles='OpenStreetMap')

    # Feature groups for toggleable layers
    fg_last_seen = folium.FeatureGroup(name="Last Known Location", show=True)
    fg_frequent = folium.FeatureGroup(name="Frequently Visited Areas", show=True)
    fg_predicted = folium.FeatureGroup(name="Predicted Search Priority Areas", show=True)
    fg_routes = folium.FeatureGroup(name="Probable Routes (Markov)", show=True)
    fg_anomalies = folium.FeatureGroup(name="Anomalous Locations", show=True)
    fg_heatmap = folium.FeatureGroup(name="Historical Trajectory Heatmap", show=False)

    # 1. Last Known Location Marker
    folium.Marker(
        location=[center_lat, center_lon],
        popup=f"<b>LAST KNOWN LOCATION</b><br>Lat: {center_lat:.4f}<br>Lon: {center_lon:.4f}",
        tooltip="Last Known Location",
        icon=folium.Icon(color='red', icon='user-secret', prefix='fa')
    ).add_to(fg_last_seen)
    
    # Pulse circle around last known location
    folium.Circle(
        location=[center_lat, center_lon],
        radius=500,
        color='red',
        fill=True,
        fill_opacity=0.2,
        popup="Primary Search Radius (500m)"
    ).add_to(fg_last_seen)

    # 2. Frequently Visited Areas
    if frequent_areas:
        for area in frequent_areas:
            lat = area['centroid_lat']
            lon = area['centroid_lon']
            name = area['area_name']
            visits = area['visit_count']
            folium.CircleMarker(
                location=[lat, lon],
                radius=10 + min(visits // 10, 15),
                color='blue',
                fill=True,
                fill_color='cyan',
                fill_opacity=0.6,
                popup=f"<b>Frequent Area: {name}</b><br>Historical Visits: {visits}",
                tooltip=f"Frequent: {name}"
            ).add_to(fg_frequent)

    # 3. Predicted Target Areas & Priority Ranking
    if predicted_rankings:
        for item in predicted_rankings:
            lat = item['lat']
            lon = item['lon']
            name = item['area_name']
            rank = item['rank']
            prob = item.get('prob_percent', 'N/A')
            priority = item.get('priority', 'Medium')
            score = item.get('score', 50.0)
            color = get_priority_color(priority)

            html_popup = f"""
            <div style="font-family: sans-serif; min-width: 180px;">
                <h4 style="margin: 0; color: {color};">Rank #{rank}: {name}</h4>
                <hr style="margin: 4px 0;">
                <b>Priority:</b> {priority}<br>
                <b>Search Priority Score:</b> {score}/100<br>
                <b>ML Probability:</b> {prob}<br>
                <b>Lat/Lon:</b> {lat:.4f}, {lon:.4f}
            </div>
            """
            
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(html_popup, max_width=250),
                tooltip=f"Rank #{rank} - {name} ({priority})",
                icon=folium.Icon(color=color, icon='search', prefix='fa')
            ).add_to(fg_predicted)

            folium.Circle(
                location=[lat, lon],
                radius=800,
                color=color,
                weight=2,
                dash_array='5, 5',
                fill=True,
                fill_opacity=0.15
            ).add_to(fg_predicted)

    # 4. Probable Route Polylines
    if probable_route_coords and len(probable_route_coords) > 1:
        folium.PolyLine(
            locations=probable_route_coords,
            color='darkpurple',
            weight=4,
            opacity=0.8,
            dash_array='8, 8',
            tooltip="Probable Movement Route Sequence"
        ).add_to(fg_routes)
        
        for idx, (rlat, rlon) in enumerate(probable_route_coords):
            folium.CircleMarker(
                location=[rlat, rlon],
                radius=5,
                color='purple',
                fill=True,
                popup=f"Route Waypoint #{idx+1}"
            ).add_to(fg_routes)

    # 5. Anomalous Locations
    if anomalous_points is not None and not anomalous_points.empty:
        for _, row in anomalous_points.iterrows():
            alat = row['latitude']
            alon = row['longitude']
            reason = row.get('anomaly_reason', 'Unusual GPS observation')
            folium.Marker(
                location=[alat, alon],
                popup=f"<b>ANOMALY DETECTED</b><br>Reason: {reason}",
                tooltip="Anomalous Point",
                icon=folium.Icon(color='black', icon='exclamation-triangle', prefix='fa')
            ).add_to(fg_anomalies)

    # 6. Historical Trajectory Heatmap
    if raw_trajectories_df is not None and not raw_trajectories_df.empty:
        heat_data = raw_trajectories_df[['latitude', 'longitude']].values.tolist()
        HeatMap(heat_data, radius=12, blur=15, max_zoom=13).add_to(fg_heatmap)

    # Add all feature groups to map
    fg_last_seen.add_to(m)
    fg_frequent.add_to(m)
    fg_predicted.add_to(m)
    fg_routes.add_to(m)
    fg_anomalies.add_to(m)
    fg_heatmap.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
