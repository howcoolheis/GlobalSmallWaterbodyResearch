import ee
import datetime

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

# Load global surface water data (as ImageCollection rather than Image)
gsw = ee.ImageCollection(GSW_ID)
print('GSW ImageCollection loaded; printing info:')
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
print('Grid created; number of cells (approx):', grid_fc.size().getInfo())

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
        scale=30,
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

# Process all years and store results
results = {}
for y in range(YEAR_START, YEAR_END + 1):
    fc = delineate_sw_for_year(y)
    if fc:
        print(f'Year {y} feature count (server-side):', fc.size().getInfo())
        results[y] = fc

print('Processing loop complete.')

# Compute grid stats for a given year
def compute_grid_stats_for_year(fc_year):
    def attach_to_cell(cell):
        lon0 = cell.getNumber('lon0')
        lat0 = cell.getNumber('lat0')
        lon_idx = ee.Number(lon0).floor()
        lat_idx = ee.Number(lat0).floor()
        feats_in_cell = fc_year.filter(ee.Filter.And(ee.Filter.eq('lon_idx', lon_idx), ee.Filter.eq('lat_idx', lat_idx)))
        np_count = feats_in_cell.size()
        mean_area = ee.Algorithms.If(np_count.gt(0), feats_in_cell.aggregate_mean('area_m2'), 0)
        return cell.set({'NP': np_count, 'AREAMN': mean_area})

    grid_stats = grid_fc.map(attach_to_cell)
    return grid_stats

# Compute grid stats for the first year (2001)
example_year = YEAR_START
grid_stats_2001 = compute_grid_stats_for_year(results[example_year])
print('Grid stats (sample) size:', grid_stats_2001.size().getInfo())

print('Script finished.')
