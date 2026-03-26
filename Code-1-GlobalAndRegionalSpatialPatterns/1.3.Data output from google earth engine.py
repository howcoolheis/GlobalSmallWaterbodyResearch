import ee
import time  # To check task completion

# Authenticate and initialize Earth Engine
try:
    ee.Initialize()
except Exception as e:
    print("Earth Engine not initialized. Authenticating...")
    ee.Authenticate()  # Prompt for authentication
    ee.Initialize()

# Define constants for data (example folder and file names)
EXPORT_FOLDER = 'GEE_exports'  # Folder in Google Drive
EXPORT_PREFIX = 'SW_metrics_all_years'  # File name prefix

# Initialize the all_metrics list to collect all the FeatureCollections
all_metrics = []

# Load the exported files from Google Drive or Earth Engine Assets
# Example of reading the exported data from Google Drive (GeoJSON format)
try:
    # Replace the paths below with the actual paths where your data is stored
    abundance_fc = ee.FeatureCollection(f'users/WenjunChen/{EXPORT_FOLDER}/{EXPORT_PREFIX}_Abundance.geojson')
    mean_area_fc = ee.FeatureCollection(f'users/WenjunChen/{EXPORT_FOLDER}/{EXPORT_PREFIX}_Areamn.geojson')
    aggregation_fc = ee.FeatureCollection(f'users/WenjunChen/{EXPORT_FOLDER}/{EXPORT_PREFIX}_Aggregation.geojson')
    connectivity_fc = ee.FeatureCollection(f'users/WenjunChen/{EXPORT_FOLDER}/{EXPORT_PREFIX}_Connectivity.geojson')

    # Append the loaded FeatureCollections to the all_metrics list
    all_metrics.append(abundance_fc)
    all_metrics.append(mean_area_fc)
    all_metrics.append(aggregation_fc)
    all_metrics.append(connectivity_fc)

    print(f"Loaded {len(all_metrics)} FeatureCollections from the exported data.")
except Exception as e:
    print(f"Error loading the exported data: {e}")

# Create an empty FeatureCollection to merge all the individual ones into one
fc_all = ee.FeatureCollection([])

# Merge all FeatureCollections in the all_metrics list
for fc in all_metrics:
    fc_all = fc_all.merge(fc)

# Define the selectors for the properties to export
selectors = ['year', 'lon0', 'lat0']
for prefix in ['NP_c', 'AREAMN_c', 'AGGREGATION_c', 'CONNECTIVITY_c']:
    for i in range(5):
        selectors.append(f'{prefix}{i}')

# Check the size of the feature collection (this will give an approximate count)
print('Server-side feature count (approx):', fc_all.size().getInfo())

# Check the property names of the first feature (example properties in the feature)
print('Example feature property names (server-side):', fc_all.first().propertyNames().getInfo())

# Function to create and start the export task
def export_to_drive(feature_collection, export_folder, file_prefix, selectors, file_format='CSV'):
    try:
        task = ee.batch.Export.table.toDrive(
            collection=feature_collection,
            description=f'{file_prefix}_export',
            folder=export_folder,  # Google Drive folder where the file will be saved
            fileNamePrefix=file_prefix,  # Prefix for the exported file
            fileFormat=file_format,  # Export format (CSV/GeoJSON/etc.)
            selectors=selectors  # Select specific columns to export
        )

        task.start()

        # Wait for the task to finish and notify the user
        while task.active():
            print("Export task is still running...")
            time.sleep(10)  # Sleep for 10 seconds before checking again

        print(f"Export task {file_prefix}_export completed successfully.")
    except Exception as e:
        print(f"Error during export task: {e}")

# Start the export task for the merged FeatureCollection
export_to_drive(fc_all, EXPORT_FOLDER, EXPORT_PREFIX, selectors, file_format='CSV')
