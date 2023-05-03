#!/usr/bin/env python3
"""Compute a risk for TB from satellite images."""

import pathlib
from functools import partial

import cv2
import geopandas as geopd
import mercantile
import numpy as np
import pyproj
import rasterio
import shapely
from PIL import Image
from rasterio import features as riofeats
from rasterio.transform import Affine
from scipy import ndimage
from shapely import geometry, ops
from skimage import color

ZOOM = 17


def find_cities(
    popcount_path: str, threshold=6000, index: int = 0, outfile: str = "cities.geojson"
):
    """Find locations at high density of population.

    Parameters
    ----------
    popcount_path : str
        path to the population count geotiff
    threshold : float
        number of persons per square km above which we determine a urban area.
    index : int
        index of the geotiff layer
    cities : str
        file name to save the cities boundaries

    Return
    ------
    cities : geopd.GeoDataFrame
        a collection in which each city is a feature.
    """
    with rasterio.open(popcount_path, mode="r") as raster:
        bounds = np.array(list(raster.bounds))
        popcount = np.asarray(raster.read()[index])
    popcount = np.ma.masked_less_equal(popcount, 0.0, copy=True)

    # find areas with pop count above threshold
    lthreshold = area(bounds) * threshold / popcount.size / 1000000
    lpop = np.empty_like(popcount)
    lpop[popcount < lthreshold] = 0
    lpop[popcount >= lthreshold] = 1

    # extend a bit to the surroundings
    lpop = riofeats.sieve(lpop.astype(rasterio.int16), int(10000 / lthreshold))
    lpop = ndimage.binary_fill_holes(lpop)
    lpop = ndimage.binary_dilation(lpop, structure=np.ones((5, 5)))

    transform = rasterio.transform.from_bounds(*bounds, *lpop.shape[::-1])
    shapes = rasterio.features.shapes(lpop.astype(rasterio.int16), transform=transform)

    features = geopd.GeoDataFrame.from_features(
        {
            "type": "FeatureCollection",
            "features": [
                {"geometry": shapely.geometry.shape(shape), "properties": {"name": f"city_{i}"}}
                for i, (shape, value) in enumerate(shapes)
                if value > 0.0
            ],
        }
    )
    features.to_file(outfile, driver="GeoJSON")


def merge_tiles(
    cities_file: str, tilename_fmt: str, outfile_fmt: int = "city_{}.geotif", zoom: int = ZOOM
):
    """Build the map from the tiles.

    Parameters
    ----------
    cities_file : str
        path to the file holding the cities contours as feature collection
    tilename_fmt : str
        path to tiles with formatting `path/to/tiles_{x}_{y}_{z}.png`
    outfile_fmt : str
        path for writing the city geotiff, it will be numbered following the cities .
        Default to `city_{}.geotif`
    zoom : int
        zoom level (optional)
    """
    cities = geopd.GeoDataFrame.from_file(cities_file)

    for city in cities.iterfeatures(show_bbox=True, drop_id=True):
        city_bbox = city["bbox"]

        # TODO: filter out tiles outside the city feature.

        # tiles numbering is;
        # x: left -> right
        # y: up -> down

        # get the lower left and upper right
        tile_ll = mercantile.tile(city_bbox[0], city_bbox[1], zoom)
        tile_ur = mercantile.tile(city_bbox[2], city_bbox[3], zoom)

        # shape in pixels
        width = (tile_ur.x - tile_ll.x + 1) * 256
        height = (tile_ll.y - tile_ur.y + 1) * 256
        image = np.zeros((height, width, 3))

        for tile in mercantile.tiles(*city_bbox, zooms=zoom):
            flnm = pathlib.Path(tilename_fmt.format(x=tile.x, y=tile.y, z=zoom))
            if flnm.is_file():
                tile_img = Image.open(flnm)
                indx = (tile.x - tile_ll.x) * 256
                indy = (tile_ll.y - tile.y) * 256

                # add tile while inverting y axis
                image[indy: indy + 256, indx: indx + 256, :] = np.asarray(tile_img)[::-1, :, :]

        image = np.asarray(image, dtype=np.uint8)

        bbox_ll = mercantile.bounds(tile_ll)
        bbox_ur = mercantile.bounds(tile_ur)
        resx = (bbox_ur.east - bbox_ll.west) / width
        resy = (bbox_ur.north - bbox_ll.south) / height
        transform = Affine.translation(bbox_ll.west, bbox_ll.south) * Affine.scale(resx, resy)
        with rasterio.open(
            f"/tmp/new{city['properties']['name']}.tif",
            "w",
            driver="GTiff",
            height=image.shape[0],
            width=image.shape[1],
            count=3,
            dtype=image.dtype,
            crs="+proj=latlong",
            transform=transform,
        ) as fout:
            fout.write(image[:, :, 0], 1)
            fout.write(image[:, :, 1], 2)
            fout.write(image[:, :, 2], 3)


def satellite(imagery: str, output: str):
    """Compute the risk for a given city."""
    with rasterio.open(imagery, mode="r") as raster:
        bounds = np.array(list(raster.bounds))
        profile = raster.profile
        satimage = np.moveaxis(np.asarray(raster.read()), 0, -1)

    sat_filter = get_sat_filter(satimage, bounds)
    profile.update({
        'nodata': 0,
        "count": 1,
        "dtype": np.float32
    })

    with rasterio.open(output, "w", **profile) as dst:
        dst.write(sat_filter, indexes=1)


def area(polygon, coords="latlong"):
    """Return the area of a given shape in square meters."""
    west, south, east, north = polygon
    square = geometry.Polygon(
        [
            [west, south],
            [west, north],
            [east, north],
            [east, south],
            [west, south],
        ]
    )
    proj = partial(
        pyproj.transform,
        pyproj.Proj(proj=coords),
        pyproj.Proj(proj="aea", lat_1=square.bounds[1], lat_2=square.bounds[3], datum="WGS84"),
    )

    return ops.transform(proj, square).area


def canny(fig):
    """Compute the canny filter."""
    b_and_w = color.rgb2gray(fig)

    # strech values to the [0, 255] interval
    b_and_w = np.interp(b_and_w, [b_and_w.min(), b_and_w.max()], [0, 255])

    edges = cv2.Canny(b_and_w.astype("uint8"), 150, 200)
    return edges


def green_hsl(img):
    """Find green throuh hsl decomposition."""
    # blur a bit averaging
    init_img = cv2.filter2D(img[:, :, ::-1], -1, np.ones((3, 3)) / 9)  # convert to BGR
    # convert to hsv
    hsv = cv2.cvtColor(init_img, cv2.COLOR_BGR2HSV)

    hsv_min = (15, 0, 0)  # black
    hsv_max = (120, 255, 120)  # light green
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    # Replace small polygons in source with value of their largest neighbor.
    mask = rasterio.features.sieve(
        mask.astype(rasterio.int16),
        50,
    )

    return 255 - mask


def filtr(density, green):
    """Combine filters on green and dense areas."""
    kernel = np.ones((41, 41)) / (41**2)

    grn = cv2.filter2D(green, -1, kernel).astype(float) / 255.0
    grn[grn < 0.5] = 0

    dens = cv2.filter2D(density, -1, kernel).astype(float) / 255.0
    dens = dens * grn

    pdens = 0.2
    # return np.clip(0, 1, (dens / pdens)**2)
    return np.clip(0, 1, dens / pdens)


def get_sat_filter(img, bounds, mask=None):
    print("Computing the canny filter")
    mat_filter = canny(img)
    print("Computing the green filter")
    mat_green = green_hsl(img)
    return filtr(mat_filter, mat_green)
