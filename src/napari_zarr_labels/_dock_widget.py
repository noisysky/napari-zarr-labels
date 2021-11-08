"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
import os

from napari_plugin_engine import napari_hook_implementation
import numpy as np
from qtpy.QtWidgets import QWidget, QGridLayout, QPushButton
import zarr
from napari.layers.labels._labels_key_bindings import activate_paint_mode
from napari.layers.labels._labels_utils import indices_in_shape, sphere_indices
from napari.layers.labels import Labels
from napari.layers.image import Image


class ZarrLabelsLayer(QWidget):
    def __init__(self, napari_viewer):
        self.viewer = napari_viewer
        super().__init__()

        layout = QGridLayout()
        create_btn = QPushButton('Create', self)
        paint_button = QPushButton('Paint', self)
        threshold_button = QPushButton('Threshold', self)
        save_button = QPushButton('Save As Zarr', self)
        self.labels_layer_object = None
        try:
            self.image_layer_objects = list(filter(lambda x: type(x) == Image, napari_viewer.layers))
        except StopIteration:
            self.image_layer_objects = None

        if self.image_layer_objects:
            self.source_dir = os.path.dirname(self.image_layer_objects[0].source.path)
            zarr_folder = os.path.join(self.source_dir, 'threshold')
            self.labels_layer_store = os.path.join(zarr_folder, 'local.zarr')
            if not os.path.exists(zarr_folder):
                os.mkdir(zarr_folder)
            self.chunks = self.image_layer_objects[0].data[0].chunksize
            self.dtype = self.image_layer_objects[0].data[0].dtype
        else:
            self.source_dir = os.getcwd()
            self.chunks = (128, 128)
            self.dtype = int

        def trigger_create():
            extent = napari_viewer.layers.extent.world
            scale = napari_viewer.layers.extent.step
            scene_size = extent[1] - extent[0]
            corner = extent[0] + 0.5 * napari_viewer.layers.extent.step
            shape = [
                np.round(s / sc).astype('int') if s > 0 else 1
                for s, sc in zip(scene_size, scale)
            ]
            empty_labels = zarr.zeros(shape, chunks=self.chunks, dtype=self.dtype)
            self.labels_layer_object = napari_viewer.add_labels(empty_labels, translate=np.array(corner), scale=scale)

        def trigger_paint():
            Labels.paint = paint  # replace native paint method with the one defined below (to work with zarr array)
            if self.labels_layer_object:
                activate_paint_mode(self.labels_layer_object)

        def trigger_threshold():
            import dask.array as da
            image_layer_object = self.image_layer_objects[0]
            array_mean = float(da.mean(image_layer_object.data[0]))
            array_stddev = float(da.std(image_layer_object.data[0]))
            mask_array = image_layer_object.data[0] > (array_mean + 3 * array_stddev)
            da.to_zarr(mask_array, self.labels_layer_store, overwrite=True)
            z1 = zarr.open_array(self.labels_layer_store, mode='a', chunks=self.chunks, dtype=self.dtype)
            z2 = zarr.array(z1, chunks=self.chunks, dtype=self.dtype)
            self.labels_layer_object.data = z2

        def trigger_save():
            zarr.save(self.labels_layer_store, self.labels_layer_object.data)

        create_btn.clicked.connect(trigger_create)
        paint_button.clicked.connect(trigger_paint)
        threshold_button.clicked.connect(trigger_threshold)
        save_button.clicked.connect(trigger_save)
        layout.addWidget(create_btn)
        layout.addWidget(paint_button)
        layout.addWidget(threshold_button)
        layout.addWidget(save_button)

        # activate layout
        self.setLayout(layout)


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return ZarrLabelsLayer


def paint(self, coord, new_label, refresh=True):
    """Paint over existing labels with a new label, using the selected
    brush shape and size, either only on the visible slice or in all
    n dimensions.

    Parameters
    ----------
    coord : sequence of int
        Position of mouse cursor in image coordinates.
    new_label : int
        Value of the new label to be filled in.
    refresh : bool
        Whether to refresh view slice or not. Set to False to batch paint
        calls.
    """
    shape = self.data.shape
    dims_to_paint = sorted(self._dims_order[-self.n_edit_dimensions :])
    dims_not_painted = sorted(self._dims_order[: -self.n_edit_dimensions])
    paint_scale = np.array(
        [self.scale[i] for i in dims_to_paint], dtype=float
    )

    slice_coord = [int(np.round(c)) for c in coord]
    if self.n_edit_dimensions < self.ndim:
        coord_paint = [coord[i] for i in dims_to_paint]
        shape = [shape[i] for i in dims_to_paint]
    else:
        coord_paint = coord

    # Ensure circle doesn't have spurious point
    # on edge by keeping radius as ##.5
    radius = np.floor(self.brush_size / 2) + 0.5
    mask_indices = sphere_indices(radius, tuple(paint_scale))

    mask_indices = mask_indices + np.round(np.array(coord_paint)).astype(
        int
    )

    # discard candidate coordinates that are out of bounds
    mask_indices = indices_in_shape(mask_indices, shape)

    # Transfer valid coordinates to slice_coord,
    # or expand coordinate if 3rd dim in 2D image
    slice_coord_temp = [m for m in mask_indices.T]
    if self.n_edit_dimensions < self.ndim:
        for j, i in enumerate(dims_to_paint):
            slice_coord[i] = slice_coord_temp[j]
        for i in dims_not_painted:
            slice_coord[i] = slice_coord[i] * np.ones(
                mask_indices.shape[0], dtype=int
            )
    else:
        slice_coord = slice_coord_temp

    slice_coord = tuple(slice_coord)

    # Fix indexing for xarray if necessary
    # See http://xarray.pydata.org/en/stable/indexing.html#vectorized-indexing
    # for difference from indexing numpy
    try:
        import xarray as xr

        if isinstance(self.data, xr.DataArray):
            slice_coord = tuple(xr.DataArray(i) for i in slice_coord)
    except ImportError:
        pass

    # slice coord is a tuple of coordinate arrays per dimension
    # subset it if we want to only paint into background/only erase
    # current label
    if self.preserve_labels:
        if new_label == self._background_label:
            keep_coords = self.data[slice_coord] == self.selected_label
        else:
            keep_coords = self.data[slice_coord] == self._background_label
        slice_coord = tuple(sc[keep_coords] for sc in slice_coord)

    if isinstance(self.data, zarr.Array):
        # save the existing values to the history
        self._save_history(
            (
                slice_coord,
                np.array(self.data.vindex[slice_coord], copy=True),
                new_label,
            )
        )
        # update the labels image
        self.data.vindex[slice_coord] = new_label
    else:
        # save the existing values to the history
        self._save_history(
            (
                slice_coord,
                np.array(self.data[slice_coord], copy=True),
                new_label,
            )
        )

        # update the labels image
        self.data[slice_coord] = new_label

    if refresh is True:
        self.refresh()
