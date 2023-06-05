from netCDF4 import Dataset

from mpl_toolkits import basemap
from pyhdf.SD import SD
from pylab import *

def read_global(cru_file, gpcc_file, clay_file, bd_file, soc_file, ph_file, Cropcomb_file, land_file, silt_file, sand_file):
    # print('opening file: ' + str(cru_file))
    cru_datafile = Dataset(cru_file)
    cru_data = cru_datafile.variables['tmp'][:].mean(axis=0)
    cru_lats = cru_datafile.variables['lat'][:]
    cru_lons = cru_datafile.variables['lon'][:]
    cru_datafile.close()
    #
    lats_common = cru_lats.copy()
    lons_common = cru_lons.copy()
    #
    del cru_lats, cru_lons

    ###########################################################################
    # common mesh grid (0.5 x 0.5)
    nlons, nlats = np.meshgrid(lons_common, lats_common)

    # gpcc_filename = '.\global_data\MAP\\normals_1991_2020_v2022_05.nc'
    # print(' opening file: ' + str(gpcc_file))
    gpcc_file = Dataset(gpcc_file)
    start_date_index = 0
    gpcc_data = gpcc_file.variables['precip'][start_date_index:, ::-1, :].mean(axis=0) * 12  # (mm/yr)
    gpcc_file.close()
    #
    gpcc_data_temp = gpcc_data.copy()
    gpcc_data[:, 0:360] = gpcc_data_temp[:, 360:]  # match to (lons_common, lats_common)
    gpcc_data[:, 360:] = gpcc_data_temp[:, 0:360]
    #
    del gpcc_data_temp

    ### load the clay + silt and bulk density data from HWSD
    # print(' opening file: ' + str(clay_file))
    clayfile_t = Dataset(clay_file, format='NETCDF4')
    claylats = clayfile_t.variables['lat'][:]
    claylons = clayfile_t.variables['lon'][:]
    claymap_in_t = clayfile_t.variables['T_CLAY'][:]
    clayfile_t.close()
    #
    claymap = basemap.interp(claymap_in_t, claylons, claylats, nlons, nlats, order=1)
    ##
    del claylats, claylons, claymap_in_t

    ##
    # print(' opening file: ' + str(silt_file))
    siltfile = Dataset(silt_file, format='NETCDF4')
    siltlats = siltfile.variables['lat'][:]
    siltlons = siltfile.variables['lon'][:]
    silttmap_in = siltfile.variables['T_SILT'][:]
    siltfile.close()
    #
    siltmap = basemap.interp(silttmap_in, siltlons, siltlats, nlons, nlats, order=1)
    #
    del siltlats, siltlons, silttmap_in

    ###
    # print(' opening file: ' + str(sand_file))
    sandfile = Dataset(sand_file, format='NETCDF4')
    sandlats = sandfile.variables['lat'][:]
    sandlons = sandfile.variables['lon'][:]
    sandmap_in = sandfile.variables['T_SAND'][:]
    sandfile.close()
    #
    sandmap = basemap.interp(sandmap_in, sandlons, sandlats, nlons, nlats, order=1)
    #
    del sandlats, sandlons, sandmap_in

    # bdfilename = ".\global_data\T_BULK_DEN.nc4"
    # print(' opening file: ' + str(bd_file))
    bdfile = Dataset(bd_file, format='NETCDF4')
    bdlats = bdfile.variables['lat'][:]
    bdlons = bdfile.variables['lon'][:]
    bdtmap_in = bdfile.variables['T_BULK_DEN'][:]
    bdfile.close()
    #
    bdmap = basemap.interp(bdtmap_in, bdlons, bdlats, nlons, nlats, order=1)
    #
    del bdlats, bdlons, bdtmap_in

    ###########################################################################
    ### load HWSD soc data
    # soctfilename = ".\global_data\T_OC.nc4"
    # print(' opening file: ' + str(soc_file))
    soctfile = Dataset(soc_file, format='NETCDF4')
    soctlats = soctfile.variables['lat'][:]
    soctlons = soctfile.variables['lon'][:]
    soctmap_in = soctfile.variables['T_OC'][:]
    soctfile.close()
    #
    socmap = basemap.interp(soctmap_in, soctlons, soctlats, nlons, nlats, order=1)

    ### calculating soil concentrations
    # socmap = socmap * 10  # into gC/kg soil
    #
    del soctmap_in
    #

    # load ph data
    # phtfilename = ".\global_data\T_PH_H2O.nc4"
    # print(' opening file: ' + str(ph_file))
    phtfile = Dataset(ph_file, format='NETCDF4')
    phlats = phtfile.variables['lat'][:]
    phlons = phtfile.variables['lon'][:]
    phtmap_in = phtfile.variables['T_PH_H2O'][:]
    phtfile.close()
    #
    phmap = basemap.interp(phtmap_in, phlons, phlats, nlons, nlats, order=1)

    del phlons, phlats, phtmap_in

    #load Cropcombination
    # cropCombinationfilename = r'D:/zcl/Cropcombination.xlsx'
    # print(' opening file: ' + str(Cropcomb_file))
    # Cropcombination = pd.read_excel(Cropcomb_file).values

    ### loading MODIS biome data
    # landfilename = ".\global_data\MCD12C1.A2021001.061.2022217040006.hdf"
    # print(' opening file: ' + str(land_file))
    landfile = SD(land_file)

    landmap_in = landfile.select("Majority_Land_Cover_Type_1").get()
    #
    landlats_fix = soctlats
    landlons_fix = soctlons
    landmap_in_fix = np.flipud(landmap_in)
    #
    landmap = basemap.interp(landmap_in_fix, landlons_fix, landlats_fix, nlons, nlats, order=0)
    #
    del landlats_fix, landlons_fix, soctlats, soctlons, landmap_in

    # initializing for grouping
    # landcover = landmap * 1
    landcover_all = landmap * 1
    old = landmap * 1  # dummy variable

    # including tundra, desert, peatland (1 to 10 categories, as in Shi et al. 2020)
    landcover_all[((old == 1) | (old == 2) | (old == 3) | (old == 4) | (old == 5) | (old == 8))  # forests boreal
                  & (nlats > 50)] = 0
    landcover_all[((old == 1) | (old == 2) | (old == 3) | (old == 4) | (old == 5) | (old == 8))  # forests temperate
                  & (nlats < 50) & (nlats > 23)] = 0
    landcover_all[((old == 1) | (old == 2) | (old == 3) | (old == 4) | (old == 5) | (old == 8))  # forests temperate
              & (nlats > -50) & (nlats < -23)] = 0
    landcover_all[((old == 1) | (old == 2) | (old == 3) | (old == 4) | (old == 5) | (old == 8))  # forests tropical
                  & (nlats < 23) & (nlats > -23)] = 0
    landcover_all[(old == 10) & (nlats < 60)] = 0  # grasslands
    landcover_all[((old == 6) | (old == 7)) & (nlats < 60)] = 0  # shrublands
    landcover_all[(old == 9)] = 0  # savannas
    landcover_all[((old == 12) | (old == 13) | (old == 14))] = 1  # cropland
    landcover_all[(old == 11)] = 0  # wetland/peatland
    landcover_all[(old == 15)] = 0  # snow/ice
    landcover_all[(old == 16)] = 0  # desert
    landcover_all[((old == 0) | (old == 17))] = 0  # water and unclassified
    landcover_all[((old == 6) | (old == 7)) & (nlats > 60)] = 0  # tundra shrubland
    landcover_all[(old == 10) & (nlats > 60)] = 0  # tundra shrubland
    #
    del old, landmap

    ### common masks between datasets
    common_mask = np.logical_or(cru_data.mask, gpcc_data.mask)
    common_mask = np.logical_or(common_mask, socmap.mask)
    common_mask = np.logical_or(common_mask, bdmap.mask)
    common_mask = np.logical_or(common_mask, phmap.mask)
    common_mask = np.ma.masked_array(common_mask, claymap)
    common_mask = np.ma.masked_array(common_mask, siltmap)
    common_mask = np.ma.masked_array(common_mask, sandmap)
    #
    # adding peatland mask to all
    soc_data = np.ma.masked_array(socmap, mask=common_mask)
    cru_data = np.ma.masked_array(cru_data, mask=common_mask)
    gpcc_data = np.ma.masked_array(gpcc_data, mask=common_mask)
    ph_data = np.ma.masked_array(phmap, mask=common_mask)
    clay_data = np.ma.masked_array(claymap, mask=common_mask)
    silt_data = np.ma.masked_array(siltmap, mask=common_mask)
    sand_data = np.ma.masked_array(sandmap, mask=common_mask)
    bd_data = np.ma.masked_array(bdmap, mask=common_mask)
    landcover = np.ma.masked_array(landcover_all, mask=common_mask)  # no mask for landcover_all
    ### flattened global datasets
    LU = landcover  # choosing  landuse to match synthesis categories
    #
    ma.set_fill_value(soc_data, 0)
    ma.set_fill_value(cru_data, 0)
    ma.set_fill_value(gpcc_data, 0)
    ma.set_fill_value(clay_data, 0)
    # ma.set_fill_value(texture_avg, 0)
    ma.set_fill_value(LU, 0)
    ma.set_fill_value(ph_data, 0)
    #
    length = np.ravel((soc_data)).shape[0]
    soc = np.ravel((soc_data)).reshape(length, 1)
    mat = np.ravel((cru_data)).reshape(length, 1)
    precip = np.ravel((gpcc_data)).reshape(length, 1)
    bd = np.ravel((bd_data)).reshape(length, 1)
    clay = np.ravel((clay_data)).reshape(length, 1)
    silt = np.ravel((silt_data)).reshape(length, 1)
    sand = np.ravel((sand_data)).reshape(length, 1)
    ph = np.ravel((ph_data)).reshape(length, 1)
    # Cropcombination = np.ravel((Cropcombination)).reshape(length, 1)

    dlatout = 0.5  # size of lat grid
    dlonout = 0.5  # size of lon grid
    outlats = np.arange(90 - dlatout / 2, -90, -dlatout)
    outlons = np.arange(-180 + dlonout / 2, 180, dlonout)

    La = np.tile(outlats, 720)
    La = np.ravel(La).reshape(length, 1)
    Lo = np.repeat(outlons, 360)
    Lo = np.ravel(Lo).reshape(length, 1)

    # bd_global = bdmap * landcover
    #
    # BD_global = pd.DataFrame(bd_global)
    # writer1 = pd.ExcelWriter('../bd.xlsx')
    # BD_global.to_excel(writer1, float_format='%.4f')
    # writer1.save()
    #
    # soctex_avg = np.asarray(np.concatenate((Lo, La, mat, precip,  bd, soc, ph, clay, silt, sand, Cropcombination), axis=1))
    soctex_avg = np.asarray(np.concatenate((Lo, La, mat, precip,  bd, soc, ph, clay, silt, sand), axis=1))
    #
    del length, soc, mat, precip, bd, clay, ph

    return soctex_avg, landcover_all
