def query_utm_crs_from_bound(bounds):
    from pyproj import CRS
    from pyproj.aoi import AreaOfInterest
    from pyproj.database import query_utm_crs_info

    utm_crs_list = query_utm_crs_info(
        datum_name='WGS 84',
        area_of_interest=AreaOfInterest(
            west_lon_degree=bounds[0],
            east_lon_degree=bounds[1],
            south_lat_degree=bounds[2],
            north_lat_degree=bounds[3],
        ),
    )
    assert len(utm_crs_list) == 1, 'The region should not stretch across UTM zones.'

    utm_crs = CRS.from_epsg(utm_crs_list[0].code)
    return utm_crs


def query_utm_transform_from_bounds(bounds):
    from pyproj import CRS, Transformer

    wgs84_crs = CRS.from_string('WGS 84')
    utm_crs = query_utm_crs_from_bound(bounds)
    to_utm = Transformer.from_crs(wgs84_crs, utm_crs, always_xy=True)

    return to_utm
