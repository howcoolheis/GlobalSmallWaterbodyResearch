import ee
import pandas as pd

# Authenticate and initialize Earth Engine
try:
    ee.Initialize()
except Exception as e:
    print("Earth Engine not initialized. Authenticating...")
    ee.Authenticate()  # Prompt for authentication
    ee.Initialize()

GSW_ID = 'JRC/GSW1_4/YearlyHistory'
COASTLINE_ASSET = 'users/WenjunChen/gshhg'
YEAR_START = 2001
YEAR_END = 2021
MIN_AREA = 1e3
MAX_AREA = 1e8
BUFFER_TOUCH = 10
GRID_MIN_LAT = -60
GRID_MAX_LAT = 80
SCALE = 30  # Scale for reduceToVectors
EXPORT_ASSETS_PREFIX = 'users/WenjunChen/SW_metrics_'

# Load global surface water dataset and coastline data
gsw = ee.ImageCollection(GSW_ID)
print('GSW ImageCollection loaded; printing info (adapt selection code if band names differ):')
print(gsw.size().getInfo())

# Load coastline data
coast_fc = ee.FeatureCollection(COASTLINE_ASSET)

# Create 1-degree grid
def make_1deg_grid(min_lon=-180, max_lon=180, min_lat=GRID_MIN_LAT, max_lat=GRID_MAX_LAT):
    lon_steps = int((max_lon - min_lon) / 1)
    lat_steps = int((max_lat - min_lat) / 1)
    polys = []
    for i in range(lon_steps):
        lon0 = min_lon + i
        lon1 = lon0 + 1
        for j in range(lat_steps):
            lat0 = min_lat + j
            lat1 = lat0 + 1
            geom = ee.Geometry.Rectangle([lon0, lat0, lon1, lat1])
            props = {'lon_index': i, 'lat_index': j, 'lon0': lon0, 'lat0': lat0}
            polys.append(ee.Feature(geom, props))
    return ee.FeatureCollection(polys)

grid_fc = make_1deg_grid()

# Get water image for a specific year
def get_year_band_image(year):
    year_str = str(year)
    filtered = gsw.filter(ee.Filter.stringContains('system:index', year_str))
    if filtered.size().gt(0):
        img = filtered.first()  # Get the first matching image
        return img.select(['waterClass']).eq(3).selfMask()
    return None  # Return null if no image is found

# Delineate small waterbodies for a given year
def delineate_sw_for_year(year):
    print(f'Processing year {year}')
    water_img = get_year_band_image(year)
    if not water_img:
        print(f'No water image found for year {year}')
        return None

    vectors = water_img.reduceToVectors(
        geometry=ee.Geometry.Rectangle([-180, GRID_MIN_LAT, 180, GRID_MAX_LAT]),
        scale=SCALE,
        geometryType='polygons',
        eightConnected=True,
        labelProperty='water',
        bestEffort=True,
        maxPixels=1e13
    )

    # Create buffer and remove coastal waterbodies
    coastline_buffer = coast_fc.geometry().buffer(100)

    def remove_coastal(feat):
        centroid = feat.geometry().centroid()
        centroid_feature = ee.Feature(centroid)  # Convert centroid to Feature
        is_coast = centroid_feature.within(coastline_buffer)  # Check if centroid is within the buffer
        return ee.Feature(feat).set('is_coast_centroid', is_coast)

    vectors_with_flag = vectors.map(remove_coastal)
    inland_vectors = vectors_with_flag.filter(ee.Filter.eq('is_coast_centroid', False))

    # Add buffer to the waterbodies to merge adjacent waterbodies
    def feature_buffered(feat):
        geom = feat.geometry()
        return ee.Feature(geom.buffer(BUFFER_TOUCH))

    buffered_union = inland_vectors.map(lambda f: ee.Feature(f.geometry().buffer(BUFFER_TOUCH))).union()
    merged_fc = ee.FeatureCollection(buffered_union)

    # Remove buffer and return the original area
    def remove_buffer(feat):
        geom = feat.geometry()
        geom_shrunk = geom.buffer(-BUFFER_TOUCH)
        return ee.Feature(geom_shrunk)

    merged_shrunk = merged_fc.map(remove_buffer)

    # Compute the area of the waterbodies
    def set_area(feat):
        a = feat.geometry().area()
        return feat.set({'area_m2': a, 'year': year})

    with_area = merged_shrunk.map(set_area)
    filtered = with_area.filter(ee.Filter.gte('area_m2', MIN_AREA)).filter(ee.Filter.lte('area_m2', MAX_AREA))

    # Add grid and classification to the waterbodies
    def set_centroid_and_grid(feat):
        ctr = feat.geometry().centroid().coordinates()
        lon = ee.Number(ctr.get(0))
        lat = ee.Number(ctr.get(1))
        lon_idx = lon.floor()
        lat_idx = lat.floor()
        area = ee.Number(feat.get('area_m2'))
        area_class = ee.Number(0)
        area_class = ee.Algorithms.If(area.gte(1e7), 4,
                                      ee.Algorithms.If(area.gte(1e6), 3,
                                                       ee.Algorithms.If(area.gte(1e5), 2,
                                                                        ee.Algorithms.If(area.gte(1e4), 1, 0))))
        return feat.set({'lon_idx': lon_idx, 'lat_idx': lat_idx, 'area_class': area_class})

    final = filtered.map(set_centroid_and_grid)
    return final

# Functions for Abundance, Mean Area, Aggregation, and Connectivity metrics
# Each function processes the grid and calculates the metrics based on the formulas provided.

# Abundance Calculation for each grid cell across the five area classes
def compute_np(fc_year):
    def stats_for_cell(cell):
        lon0 = ee.Number(cell.get('lon0'))
        lat0 = ee.Number(cell.get('lat0'))
        lon_idx = lon0.floor()
        lat_idx = lat0.floor()

        cell_out = cell
        for ac in range(5):
            feats = fc_year.filter(
                ee.Filter.And(ee.Filter.eq('lon_idx', lon_idx), ee.Filter.eq('lat_idx', lat_idx))).filter(
                ee.Filter.eq('area_class', ac))
            np_count = feats.size()
            cell_out = cell_out.set({f'NP_c{ac}': np_count})
        return cell_out

    grid_stats = grid_fc.map(stats_for_cell)
    return grid_stats

# Mean Area Calculation for each grid cell across the five area classes
def compute_areamn(fc_year):
    def stats_for_cell(cell):
        lon0 = ee.Number(cell.get('lon0'))
        lat0 = ee.Number(cell.get('lat0'))
        lon_idx = lon0.floor()
        lat_idx = lat0.floor()

        cell_out = cell
        for ac in range(5):
            feats = fc_year.filter(
                ee.Filter.And(ee.Filter.eq('lon_idx', lon_idx), ee.Filter.eq('lat_idx', lat_idx))).filter(
                ee.Filter.eq('area_class', ac))
            np_count = feats.size()
            mean_area = ee.Algorithms.If(np_count.gt(0), feats.aggregate_mean('area_m2'), 0)
            cell_out = cell_out.set({f'AREAMN_c{ac}': mean_area})
        return cell_out

    grid_stats = grid_fc.map(stats_for_cell)
    return grid_stats

# Aggregation Calculation for each grid cell across the five area classes
def compute_aggregation(fc_year):
    def cell_aggregation(cell):
        lon0 = ee.Number(cell.get('lon0'))
        lat0 = ee.Number(cell.get('lat0'))
        lon_idx = lon0.floor()
        lat_idx = lat0.floor()

        cell_out = cell
        for ac in range(5):  # Iterate through the five area classes
            # Get the water patches in the current grid cell and area class
            patches = fc_year.filter(
                ee.Filter.And(ee.Filter.eq('lon_idx', lon_idx), ee.Filter.eq('lat_idx', lat_idx))).filter(
                ee.Filter.eq('area_class', ac))
            n_patches = patches.size()

            # Count adjacencies between patches in the same area class
            adjacencies = patches.map(lambda f: f.geometry().adjacent()).size()

            # Theoretical maximum adjacencies (fully aggregated configuration)
            max_adj = n_patches * (n_patches - 1)

            # Calculate the aggregation ratio
            aggregation = ee.Algorithms.If(max_adj.gt(0), adjacencies.divide(max_adj), 0)
            cell_out = cell_out.set({f'AGGREGATION_c{ac}': aggregation})

        return cell_out

    grid_aggregation = grid_fc.map(cell_aggregation)
    return grid_aggregation

# Connectivity Calculation for each grid cell across the five area classes
def compute_cohesion(fc_year):
    def cell_cohesion(cell):
        lon0 = ee.Number(cell.get('lon0'))
        lat0 = ee.Number(cell.get('lat0'))
        lon_idx = lon0.floor()
        lat_idx = lat0.floor()

        cell_out = cell
        for ac in range(5):  # Iterate through the five area classes
            # Get the water patches in the current grid cell and area class
            feats = fc_year.filter(
                ee.Filter.And(ee.Filter.eq('lon_idx', lon_idx), ee.Filter.eq('lat_idx', lat_idx))).filter(
                ee.Filter.eq('area_class', ac))

            sum_p = feats.aggregate_sum('perimeter_m')
            sum_psqrt = feats.aggregate_sum('perimeter_m').multiply(feats.aggregate_sum('area_m2').sqrt())

            Z = feats.size()  # Number of patches in the grid cell

            # Calculate the connectivity index using the provided formula
            connectivity = (1 - (sum_p / sum_psqrt)) * (1 - (1 / (Z.sqrt()))) ** (-1) * 100
            cell_out = cell_out.set({f'CONNECTIVITY_c{ac}': connectivity})

        return cell_out

    grid_connectivity = grid_fc.map(cell_cohesion)
    return grid_connectivity

# Projecting to Mollweide equal-area projection
mollweide_projection = ee.Projection('EPSG:54009')  # Mollweide projection EPSG code
gsw = gsw.reproject(mollweide_projection)

# Function to process each year and export results
def process_year(year):
    print(f"Processing year {year}")

    results_year = delineate_sw_for_year(year)

    # Example metric calculations (Abundance, Mean Area, etc.)
    abundance_results = compute_np(results_year)
    mean_area_results = compute_areamn(results_year)
    aggregation_results = compute_aggregation(results_year)
    connectivity_results = compute_cohesion(results_year)

    # Export results to Drive
    def export_to_drive(collection, metric, year):
        task = ee.batch.Export.table.toDrive(
            collection=collection,
            description=f"SW_Results_{year}_{metric}",
            fileFormat='GeoJSON'
        )
        task.start()

    export_to_drive(abundance_results, 'Abundance', year)
    export_to_drive(mean_area_results, 'Areamn', year)
    export_to_drive(aggregation_results, 'Aggregation', year)
    export_to_drive(connectivity_results, 'Connectivity', year)

# Iterate through all years and process
for year in range(YEAR_START, YEAR_END + 1):
    process_year(year)
