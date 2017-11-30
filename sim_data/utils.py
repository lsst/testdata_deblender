import galsim
import os
from collections import OrderedDict

import yaml
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table as ApTable

def matchCatalogs(x_ref, y_ref, x_p, y_p):
    posRef = np.dstack([x_ref, y_ref])[0]
    posP = np.dstack([x_p, y_p])[0]
    mytree = scipy.spatial.cKDTree(posRef)
    dist, index = mytree.query(posP)
    return dist, index

def get_shear(q):
    theta = np.random.rand()*2*np.pi
    shear = galsim.Shear(q=q, beta=theta*galsim.radians)
    return shear

class PowerLaw:
    """Power Law distribution to use for sampling
    """
    def __init__(self, min_, max_, gamma):
        self.min = min_
        self.max = max_
        self.gamma_p1 = gamma + 1
        if self.gamma_p1 == 0:
            self.base = np.log(self.min)
            self.norm = np.log(self.max/self.min)
        else:
            self.base = np.power(self.min, self.gamma_p1);
            self.norm = np.power(self.max, self.gamma_p1) - self.base;

    def sample(self):
        v = np.random.rand() * self.norm + self.base;
        if self.gamma_p1 == 0:
            return np.exp(v)
        else:
             return np.power(v, 1./self.gamma_p1);

class Flat:
    """Sample from a flat distribution
    """
    def __init__(self, min_, max_):
        self.min = min_
        self.max = max_

    def sample(self):
        return np.random.rand()*(self.max-self.min) + self.min

class Fixed:
    """Use a fixed value instead of sampling
    """
    def __init(self, value):
        self.value = value
    def sample(self):
        return value

class SimObj:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.radius = 0
        self.sed = 0
        self.fluxes = {}
        self.is_star = False
        self.intensity = {}
        self.component = None
    def __str__(self):
        return "<SimObj>: ("+", ".join("{0}={1}".format(k,v) for k,v in self.__dict__.items())+")"
    def __repr__(self):
        return self.__str__()

class GalaxySampler:
    def __init__(self, rng, max_redshift=1.5, seds=None):
        self.rng = rng
        self.max_redshift = max_redshift
        if seds is None:
            self.seds = get_galaxy_seds()

    def generate_sed(self):
        redshift = np.random.rand()*self.max_redshift
        sed = np.random.choice(self.seds,1)[0].atRedshift(redshift)
        return sed, redshift

    def sample(self):
        raise Exception("This method must be replaced with a sampler")

class RealGalaxySampler(GalaxySampler):
    def __init__(self, rng, filename=None, filepath=None, pixel_scale=.23, max_redshift=1.5, seds=None):
        if filename is None:
            filename = "real_galaxy_catalog_23.5_example.fits"
        if filepath is None:
            filepath = "./galsim_data"
        self.real_galaxy_catalog = galsim.RealGalaxyCatalog(filename, dir=filepath)

        super().__init__(rng, max_redshift, seds)

    def sample(self, idx, px, py, flux, filter_norm, filters, psfs):
        obj = SimObj()
        obj.index = idx
        obj.x = px
        obj.y = py
        idx = np.random.randint(100)
        morphology = galsim.RealGalaxy(self.real_galaxy_catalog, index=idx, flux=flux)
        sed, redshift = self.generate_sed()
        sed = sed.withFlux(1, filters[filter_norm])
        sed = np.array([sed.calculateFlux(f) for _,f in filters.items()])
        convolved_image = {band:galsim.Convolve([morphology*sed[s], psfs[band]])
                           for s, band in enumerate(filters)}
        
        obj.index = idx
        obj.redshift = redshift
        obj.is_star = False
        obj.catalog_index = idx
        obj.component = "mix"
        obj.sed = sed/np.sum(sed)
        
        return [obj], [convolved_image]

class SimulatedGalaxySampler(GalaxySampler):
    def __init__(self, rng, bulge, disk, color_gradient, max_redshift=1.5, seds=None, scale=1):
        # Create Radial Samplers
        self.bulge_rad_dist = globals()[bulge["radius"]["class"]](**bulge["radius"]["params"])
        if disk["radius"] is None:
            self.disk_rad_dist = self.bulge_rad_dist
        else:
            self.disk_rad_dist = globals()[disk["radius"]["class"]](**disk["radius"]["params"])

        # Create Ellipticity Samplers
        self.bulge_e_dist = globals()[bulge["ellipticity"]["class"]](**bulge["ellipticity"]["params"])
        if disk["ellipticity"] is None:
            self.disk_e_dist = None
        else:
            self.disk_e_dist = globals()[disk["ellipticity"]["class"]](**disk["ellipticity"]["params"])

        # Distribution of disk fractions
        self.disk_frac_dist = globals()[disk["frac"]["class"]](**disk["frac"]["params"])

        # Create color gradient sampler
        self.color_dist = globals()[color_gradient["class"]](**color_gradient["params"])

        self.scale = scale
        self.bulge_model = bulge["model"]
        self.disk_model = disk["model"]
        self.disk_scale = disk["min_ratio"]

        super().__init__(rng, max_redshift, seds)

    def sample(self, idx, px, py, flux, filter_norm, filters, psfs):
        # Create the bulge
        bulge_radius = self.bulge_rad_dist.sample()*self.scale
        kwargs = self.bulge_model["params"]
        kwargs[self.bulge_model["radius"]] = bulge_radius
        bulge = getattr(galsim, self.bulge_model["class"])(**kwargs)
        bulge_shear = get_shear(self.bulge_e_dist.sample())
        bulge = bulge.shear(bulge_shear)

        # Create the disk
        disk_radius = 0
        while disk_radius < self.disk_scale*bulge_radius:
            disk_radius = self.disk_rad_dist.sample()*self.scale*self.disk_scale
        kwargs = self.disk_model["params"]
        kwargs[self.disk_model["radius"]] = disk_radius
        disk = getattr(galsim, self.disk_model["class"])(**kwargs)
        # Use the bulge shear unless a different disk shear distribution is specified
        if self.disk_e_dist is not None:
            disk_shear = get_shear(self.disk_e_dist.sample())
        else:
            disk_shear = bulge_shear
        disk = disk.shear(disk_shear)

        # Combine the bulge and disk
        disk_frac = self.disk_frac_dist.sample()
        radius = (1-disk_frac)*bulge_radius + disk_frac*disk_radius
        bulge = (1-disk_frac)*bulge
        disk = disk_frac*disk

        sed, redshift = self.generate_sed()
        sed = sed.withFlux(1.0, filters[filter_norm])
        bulge_sed = np.array([sed.calculateFlux(f) for _,f in filters.items()])
        #disk_sed = np.linspace(-0.015, 0.015, len(bulge_sed))
        color_offset = self.color_dist.sample()
        disk_sed = np.linspace(-color_offset, color_offset, len(bulge_sed))
        disk_sed = bulge_sed-disk_sed

        components = [bulge, disk]
        seds = [flux*bulge_sed, flux*disk_sed]
        fractions = [1-disk_frac, disk_frac]

        objs = []
        convolved_images = []
        # Package the results
        for n, (morphology, sed) in enumerate(zip(components, seds)):
            obj = SimObj()
            obj.index = idx
            obj.x = px
            obj.y = py
            obj.redshift = redshift
            obj.is_star = False
            obj.bulge_q = 1/np.exp(bulge_shear.eta)
            obj.bulge_theta = bulge_shear.beta.rad()
            obj.disk_q = 1/np.exp(disk_shear.eta)
            obj.disk_theta = disk_shear.beta.rad()
            obj.bulge_radius = bulge_radius
            obj.disk_radius = disk_radius
            obj.radius = radius
            obj.flux_fraction = fractions[n]
            obj.sed = sed/np.sum(sed)
            obj.color_offset = color_offset
            if n == 0:
                obj.component = "bulge"
            elif n==1:
                obj.component = "disk"
            elif n==3:
                obj.component = "sfm"
            else:
                err = "Something unexpected happend, received more than 3 components of this galaxy!"
                raise ValueError(err)
            objs.append(obj)
            cimg = {band:galsim.Convolve([morphology*sed[s], psfs[band]]) for s, band in enumerate(filters)}
            convolved_images.append(cimg)

        return objs, convolved_images

def get_stellar_seds(pickles_path, spec_types, target_flux_density=1.0, wavelength=500, show=False):
    """Create templates for stellar sources
    """
    pickles_idx = ApTable.read(os.path.join(pickles_path, "pickles_uk.fits"))
    pickles_spt = [spt.rstrip() for spt in pickles_idx["SPTYPE"]]
    
    seds_stellar = []
    for spt in spec_types:
        filename = os.path.join(pickles_path, pickles_idx[pickles_spt.index(spt)]["FILENAME"].rstrip()+".fits")
        data = fits.open(filename)[1].data
        tbl = galsim.LookupTable(data["wavelength"], data["flux"])
        sed = galsim.SED(tbl, wave_type="a", flux_type="flambda")
        seds_stellar.append(sed.withFluxDensity(target_flux_density=target_flux_density,
                                                     wavelength=wavelength))
    if show:
        for s, spt in enumerate(spec_types):
            tmp = seds_stellar[s]
            plt.plot(tmp._spec.x, tmp._spec.f/np.sum(tmp._spec.f), label=spt)
            plt.title("Stellar Spectra")
            plt.xlabel("Wavelength")
            plt.ylabel("Normalized Flux")
        plt.legend()
        plt.show()

    return seds_stellar

def get_galaxy_seds(n_templates=25, target_flux_density=1.0, wavelength=500, show=False):
    """Get an array of galaxy SED's
    """
    true_gals = []
    seds = []
    sed_names = [s for s in os.listdir(galsim.meta_data.share_dir) if s.endswith(".sed") and not "more" in s]
    for sed_name in sed_names:
        sed_filename = os.path.join(galsim.meta_data.share_dir, sed_name)
        sed = galsim.SED(sed_filename, wave_type='Ang', flux_type='flambda')
        seds.append(sed.withFluxDensity(target_flux_density=target_flux_density,
                                        wavelength=wavelength))

    templates = np.random.rand(n_templates, len(sed_names))
    # normalize contribution from each template
    templates = np.array([ a/np.sum(a) for a in templates])
    seds_galaxy = []
    for template in templates:
        sed = template[0]*seds[0]
        for v,t in zip(template[1:],seds[1:]):
            sed += v*t
        seds_galaxy.append(sed)

    if show:
        for s,sed in enumerate(seds):
            plt.plot(sed._spec.x, sed._spec.f, label=sed_names[s][:-4])
            plt.legend()
            plt.xlabel("Wavelength")
            plt.ylabel("Flux")
        plt.title("Galaxy Base Spectra")
        plt.show()

    return seds_galaxy

def get_filters(names, path, file_template, show=False):
    """Load band pass filters to generate images
    """
    # Load Band Pass Filters
    filters = OrderedDict()
    for filter_name in names:
        filter_filename = os.path.join(path, file_template.format(filter_name))
        filters[filter_name] = galsim.Bandpass(filter_filename, wave_type='nm')
        filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)

    if show:
        for f, bp in filters.items():
            plt.plot(bp._tp.x, bp._tp.f, label=f)
        plt.title("LSST Band Pass Filters")
        plt.legend()
        plt.show()

    return filters

def get_psf(rng, psf_settings):
    """Get a galsim PSF for a single band
    """
    psf = getattr(galsim, psf_settings["class"])(**psf_settings["params"])
    # Shear the psf
    _ellipticity = psf_settings["ellipticity"]
    psf_dist = globals()[_ellipticity["class"]](**_ellipticity["params"])
    theta = np.random.rand()*2*np.pi
    shear = get_shear(psf_dist.sample())
    return psf.shear(shear)

def create_blend(rng, settings, filters=None, stellar_seds=None, galaxy_sampler=None, psfs=None, noise=None):
    """Create a blend with a mixture of stars and galaxies
    """
    if galaxy_sampler is None:
        _settings = settings["galactic"]["sampler"]
        galaxy_sampler = globals()[_settings["class"]](rng, **_settings["params"])

    if noise is None:
        _settings = settings["image"]["noise"]
        noise = getattr(galsim, _settings["class"])(rng, **_settings["params"])

    if stellar_seds is None:
        stellar_seds = get_stellar_seds(show=False, **settings["stellar"]["seds"])

    if filters is None:
        filters = get_filters(**settings["filters"]["files"])

    if "scale" in settings["galactic"]["sampler"]["params"]:
        scale = settings["galactic"]["sampler"]["params"]["scale"]
    else:
        scale = settings["galactic"]["sampler"]["scale"]

    # PSF model
    if psfs is None:
        _settings = settings["psf"]["psfs"]
        if len(_settings) == 1:
            _settings = _settings[list(_settings.keys())[0]]
            _psf = get_psf(rng, _settings)
            psfs = {f: _psf for f in filters}
        else:
            psfs = {}
            for f in filters:
                psfs[f] = get_psf(rng, _settings[f])

    # Set the image size of the current blend
    _settings = settings["image"]["width"]
    width = int(globals()[_settings["class"]](**_settings["params"]).sample())
    _settings = settings["image"]["height"]
    height = int(globals()[_settings["class"]](**_settings["params"]).sample())

    # Initialize Peak Location Sampler to keep sources preferentially toward the center
    # and away form the edges
    _settings = settings["position"]
    pos_dist = PowerLaw(_settings["min_"], 0.8*np.sqrt((height/2)**2+(width/2)**2), _settings["gamma"])

    # Create Empty Images in each Band
    bounds = galsim.BoundsI(0, width, 0, height)
    inner_bounds = galsim.BoundsI(10, width-10, 10, height-10)
    center = bounds.center()
    images = OrderedDict()
    for filter_name in filters.keys():
        images[filter_name] = galsim.ImageF(bounds, scale=scale)

    # Flux ranges
    _settings = settings["flux"]
    flux_dist = globals()[_settings["class"]](**_settings["params"])

    # Add galaxies until the image is dense enough
    src_x = []
    src_y = []
    sources = []

    # Set the number of sources based on the desired density
    n_sources = int(width*height*1/settings["density"])
    if n_sources < settings["image"]["n_galaxies"]["min_"]:
        n_sources = settings["image"]["n_galaxies"]["min_"]
    elif n_sources > settings["image"]["n_galaxies"]["max_"]:
        n_sources = settings["image"]["n_galaxies"]["max_"]
    for n in range(n_sources):
        # Make sure next object is suffiently away from already inserted objects
        while True:
            r = pos_dist.sample()
            theta = np.random.rand()*2*np.pi
            offsetx = r*np.cos(theta)
            offsety = r*np.sin(theta)

            px = offsetx + center.x
            py = offsety + center.y
            
            if inner_bounds.includes(int(px),int(py)) is False:
                continue
            dist, index = matchCatalogs([px], [py], src_x, src_y)
            if np.sum(dist < settings["min_separation"]) == 0 :
                    break
        src_x.append(px)
        src_y.append(py)

        is_star = False
        if np.random.rand() < settings["stellar"]["frac"]:
            is_star = True

        # Sample from the flux distribution
        flux = flux_dist.sample()
        # Add a star
        if is_star:
            obj = SimObj()
            obj.index = n
            obj.x = px
            obj.y = py
            _settings = settings["stellar"]["morphology"]
            morphology = getattr(galsim, _settings["class"])(**_settings["params"])
            sed = np.random.choice(stellar_seds,1)[0]
            obj.is_star = True
            obj.redshift = 0
            obj.radius = 0
            obj.component = "star"
            objs = [obj]

            sed = sed.withFlux(flux, filters[settings["filters"]["norm"]])
            sed = np.array([sed.calculateFlux(f) for _,f in filters.items()])
            cimg = {band:galsim.Convolve([morphology*sed[s], psfs[band]]) for s, band in enumerate(filters)}
            cimgs = [cimg]
            obj.sed = sed/np.sum(sed)
        # Add a galaxy
        else:
            objs, cimgs = galaxy_sampler.sample(n, px, py, flux, settings["filters"]["norm"], filters, psfs)

        for m,obj in enumerate(objs):
            cimg = cimgs[m]
            for band, f in filters.items():
                tmp = cimg[band].drawImage(image=images[band], add_to_image=True, offset=(offsetx, offsety))
                obj.fluxes[band] = tmp.added_flux
                # Build the truth image for the object
                true_image = galsim.ImageF(bounds, scale=scale)
                cimg[band].drawImage(image=true_image, add_to_image=False, offset=(offsetx, offsety))
                obj.intensity[band] = true_image.array
            sources.append(obj)
    for f, img in images.items():
        img.addNoise(noise)
    return images, sources, psfs

def build_catalog(sources):
    """Build an astropy table from a list of simulated objects
    """
    columns = ["index", "x", "y", "is_star", "redshift"]
    for src in sources:
        for key in src.__dict__:
            if key not in columns:
                columns.append(key)
    columns.remove("fluxes")
    columns.remove("intensity")

    data = OrderedDict()
    for col in columns:
        data[col] = np.array([getattr(src, col) if hasattr(src,col) else np.nan for src in sources])
    for f in src.intensity.keys():
        data["intensity_"+f] = np.array([src.intensity[f] for src in sources])
        
    tbl = ApTable(data)
    return tbl

if __name__ == "__main__":
    settings = yaml.load_safe("setting.yaml")
    seed = settings["seed"]
    rng = galsim.UniformDeviate(seed)
    np.random.seed(seed)