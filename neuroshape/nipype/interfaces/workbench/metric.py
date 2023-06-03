# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
    This module provides interfaces for workbench surface commands
    Adapted from nipype.interfaces.workbench
    
    .MetricGradient, .MetricGradientInputSpec, .MetricGradientOutputSpec
    developed by Nikitas C. Koussis 2023
    
"""
import os

from nipype.interfaces.base import TraitedSpec, File, Str, traits, CommandLineInputSpec
from nipype.interfaces.workbench.base import WBCommand
import logging

iflogger = logging.getLogger("nipype.interface")

class MetricResampleInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="The metric file to resample",
    )
    current_sphere = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=1,
        desc="A sphere surface with the mesh that the metric is currently on",
    )
    new_sphere = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=2,
        desc="A sphere surface that is in register with <current-sphere> and"
        " has the desired output mesh",
    )
    method = traits.Enum(
        "ADAP_BARY_AREA",
        "BARYCENTRIC",
        argstr="%s",
        mandatory=True,
        position=3,
        desc="The method name - ADAP_BARY_AREA method is recommended for"
        " ordinary metric data, because it should use all data while"
        " downsampling, unlike BARYCENTRIC. If ADAP_BARY_AREA is used,"
        " exactly one of area_surfs or area_metrics must be specified",
    )
    out_file = File(
        name_source=["new_sphere"],
        name_template="%s.out",
        keep_extension=True,
        argstr="%s",
        position=4,
        desc="The output metric",
    )
    area_surfs = traits.Bool(
        position=5,
        argstr="-area-surfs",
        xor=["area_metrics"],
        desc="Specify surfaces to do vertex area correction based on",
    )
    area_metrics = traits.Bool(
        position=5,
        argstr="-area-metrics",
        xor=["area_surfs"],
        desc="Specify vertex area metrics to do area correction based on",
    )
    current_area = File(
        exists=True,
        position=6,
        argstr="%s",
        desc="A relevant anatomical surface with <current-sphere> mesh OR"
        " a metric file with vertex areas for <current-sphere> mesh",
    )
    new_area = File(
        exists=True,
        position=7,
        argstr="%s",
        desc="A relevant anatomical surface with <current-sphere> mesh OR"
        " a metric file with vertex areas for <current-sphere> mesh",
    )
    roi_metric = File(
        exists=True,
        position=8,
        argstr="-current-roi %s",
        desc="Input roi on the current mesh used to exclude non-data vertices",
    )
    valid_roi_out = traits.Bool(
        position=9,
        argstr="-valid-roi-out",
        desc="Output the ROI of vertices that got data from valid source vertices",
    )
    largest = traits.Bool(
        position=10,
        argstr="-largest",
        desc="Use only the value of the vertex with the largest weight",
    )
    
class MetricResampleOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="the output metric")
    roi_file = File(desc="ROI of vertices that got data from valid source vertices")

class MetricResample(WBCommand):
    """
    Resample a metric file to a different mesh

    Resamples a metric file, given two spherical surfaces that are in
    register.  If ``ADAP_BARY_AREA`` is used, exactly one of -area-surfs or
    ``-area-metrics`` must be specified.

    The ``ADAP_BARY_AREA`` method is recommended for ordinary metric data,
    because it should use all data while downsampling, unlike ``BARYCENTRIC``.
    The recommended areas option for most data is individual midthicknesses
    for individual data, and averaged vertex area metrics from individual
    midthicknesses for group average data.

    The ``-current-roi`` option only masks the input, the output may be slightly
    dilated in comparison, consider using ``-metric-mask`` on the output when
    using ``-current-roi``.

    The ``-largest option`` results in nearest vertex behavior when used with
    ``BARYCENTRIC``.  When resampling a binary metric, consider thresholding at
    0.5 after resampling rather than using ``-largest``.

    >>> from shape.nipype.interfaces.workbench import MetricResample
    >>> metres = MetricResample()
    >>> metres.inputs.in_file = 'sub-01_task-rest_bold_space-fsaverage5.L.func.gii'
    >>> metres.inputs.method = 'ADAP_BARY_AREA'
    >>> metres.inputs.current_sphere = 'fsaverage5_std_sphere.L.10k_fsavg_L.surf.gii'
    >>> metres.inputs.new_sphere = 'fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii'
    >>> metres.inputs.area_metrics = True
    >>> metres.inputs.current_area = 'fsaverage5.L.midthickness_va_avg.10k_fsavg_L.shape.gii'
    >>> metres.inputs.new_area = 'fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii'
    >>> metres.cmdline
    'wb_command -metric-resample sub-01_task-rest_bold_space-fsaverage5.L.func.gii \
    fsaverage5_std_sphere.L.10k_fsavg_L.surf.gii \
    fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii \
    ADAP_BARY_AREA fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.out \
    -area-metrics fsaverage5.L.midthickness_va_avg.10k_fsavg_L.shape.gii \
    fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii'
    """

    input_spec = MetricResampleInputSpec
    output_spec = MetricResampleOutputSpec
    _cmd = "wb_command -metric-resample"

    def _format_arg(self, opt, spec, val):
        if opt in ["current_area", "new_area"]:
            if not self.inputs.area_surfs and not self.inputs.area_metrics:
                raise ValueError(
                    "{} was set but neither area_surfs or"
                    " area_metrics were set".format(opt)
                )
        if opt == "method":
            if (
                val == "ADAP_BARY_AREA"
                and not self.inputs.area_surfs
                and not self.inputs.area_metrics
            ):
                raise ValueError(
                    "Exactly one of area_surfs or area_metrics" " must be specified"
                )
        if opt == "valid_roi_out" and val:
            # generate a filename and add it to argstr
            roi_out = self._gen_filename(self.inputs.in_file, suffix="_roi")
            iflogger.info("Setting roi output file as", roi_out)
            spec.argstr += " " + roi_out
        return super(MetricResample, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = super(MetricResample, self)._list_outputs()
        if self.inputs.valid_roi_out:
            roi_file = self._gen_filename(self.inputs.in_file, suffix="_roi")
            outputs["roi_file"] = os.path.abspath(roi_file)
        return outputs

class MetricGradientInputSpec(CommandLineInputSpec):
    surface_in = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="The surface file to compute the gradient on",
    )
    metric_in = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=1,
        desc="The metric to compute the gradient of",
    )
    metric_out = File(
        name_source=["metric_in"],
        name_template="%s.out",
        keep_extension=True,
        mandatory=True,
        argstr="%s",
        position=2,
        desc="The magnitude of the gradient",
    )
    presmooth = traits.Bool(
        position=3,
        argstr="-presmooth %s",
        desc="Smooth metric before computing the gradient",
    )
    fwhm = traits.Bool(
        position=4,
        argstr="-fwhm",
        desc="Kernel size is FWHM, not sigma",
        )
    roi = traits.Bool(
        position=5,
        argstr="-roi",
        desc="Select a region of interest to take the gradient of",
        )
    roi_metric = File(
        position=6,
        exists=True,
        argstr="%s",
        desc="The area to take the gradient within, as a metric",
        )
    match_columns = traits.Bool(
        position=7,
        argstr="-match-columns",
        desc="For each input column, use the corresponding column from the roi",      
        )
    vectors = traits.Bool(
        position=8,
        argstr="-vectors",
        desc="Output gradient vectors",
        )
    vector_out = File(
        name_source=["metric_in"],
        name_template="%s_vector.out",
        keep_extension=True,
        position=9,
        argstr="%s",
        desc="The vectors as a metric file",
        )
    column = traits.Bool(
        position=10,
        argstr="-column %s",
        desc="Select a single column to compute the gradient of",
        )
    corrected_areas = traits.Bool(
        position=11,
        argstr="-corrected-areas",
        desc="Vertex areas to use instead of computing them from the surface",
        )
    area_metric = File(
        position=12,
        name_source=["metric_in"],
        name_template="%s_corrected_areas.out",
        keep_extension=True,
        argstr="%s",
        desc="The corrected area vertex areas, as a metric",
        )
    average_normals = traits.Bool(
        position=13,
        argstr="-average-normals",
        desc="Average the normals of each vertex with its neighbors before using them to compute the gradient",
        )
    
class MetricGradientOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="the output metric")
    vector_file = File(desc="the vectors of the gradient as an output file")
    corrected_metric = File(desc="the corrected vertex areas, as a metric")
    
class MetricGradient(WBCommand):
    """
    Surface gradient of a metric file
    
    At each vertex, the immediate neighbors are unfolded onto a plane tangent
    to the surface at the vertex (specifically, perpendicular to the normal).
    The gradient is computed using a regression between the unfolded
    positions of the vertices and their values.  The gradient is then given
    by the slopes of the regression, and reconstructed as a 3D gradient
    vector.  By default, takes the gradient of all columns, with no
    presmoothing, across the whole surface, without averaging the normals of
    the surface among neighbors.

    When using -corrected-areas, note that it is an approximate correction.
    Doing smoothing on individual surfaces before averaging/gradient is
    preferred, when possible, in order to make use of the original surface
    structure.

    Specifying an ROI will restrict the gradient to only use data from where
    the ROI metric is positive, and output zeros anywhere the ROI metric is
    not positive.

    By default, the first column of the roi metric is used for all input
    columns.  When -match-columns is specified to the -roi option, the input
    and roi metrics must have the same number of columns, and for each input
    column's index, the same column index is used in the roi metric.  If the
    -match-columns option to -roi is used while the -column option is also
    used, the number of columns of the roi metric must match the input
    metric, and it will use the roi column with the index of the selected
    input column.

    The vector output metric is organized such that the X, Y, and Z
    components from a single input column are consecutive columns.
    
    >>> from shape.nipype.interfaces.workbench import MetricGradient
    >>> grads = MetricGradient()
    >>> 
    
    """
    input_spec = MetricGradientInputSpec
    output_spec = MetricGradientOutputSpec
    _cmd = "wb_command -metric-gradient"

    def _format_arg(self, opt, spec, val):
        if opt == "kernel":
            if (
                type(val) == float or type(val) == int
                ):
                raise ValueError(
                    "Kernel must be given in mm"
                    )
        if opt == "fwhm":
            if not self.inputs.presmooth:
                raise ValueError(
                    "{} was set but presmooth was not set".format(opt)
                    )
        if opt == "roi":
            if not self.inputs.roi_metric:
                raise ValueError(
                    "{} was set but roi_metric was not set".format(opt)
                    )
        if opt == "match_columns":
            if not self.inputs.roi or not self.inputs.roi_metric:
                raise ValueError(
                    "{} was set but roi or roi_metric were not set".format(opt)
                    )
        if opt == "vectors":
            if not self.inputs.vector_out:
                raise ValueError(
                    "{} was set but vector_out was not set".format(opt)
                    )
        if opt == "corrected_areas":
            if not self.inputs.area_metric:
                raise ValueError(
                    "{} was set but area_metric was not set".format(opt))
        return super(MetricResample, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = super(MetricResample, self)._list_outputs()
        return outputs