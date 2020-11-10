'''
    Saving files in VTK format
'''


__all__ = ['points_vtk']


import os
import warnings
import copy

import xml.etree.ElementTree as ET

import numpy as np


def np_to_vtk_dtype(np_dtype):
    '''
        Wrapper for a dictionary.
    '''

    np_to_vtk_dtype = {
            'int8': "Int8",
            'uint8': "UInt8",
            'int16': 'Int16',
            'uint16': 'UInt16',
            'int32': 'Int32',
            'uint32': 'UInt32',
            'int64': 'Int64',
            'uint64': 'UInt64',
            'float32': 'Float32',
            'float64': 'Float64' }

    return np_to_vtk_dtype[np_dtype]


def to_unicode(numpy_array, **kwargs):
    '''
        Transform numpy_array into unicode string.
        Multidimensional arrays are transformed to unidemnisonal
        using np.ndarray.ravel which uses by default the C - type ordering.

        Arguments other than the numpy_array as passed to numpy_array.ravel()

    '''
    # Ravel the array using C - type  ordering
    if len(numpy_array.shape) > 1:
        if numpy_array.shape[1] > 1:
            numpy_array_raveled = numpy_array.ravel(**kwargs)
        else:
            numpy_array_raveled = numpy_array
    else:
        numpy_array_raveled = numpy_array

    return ' '.join(a for a in numpy_array_raveled.astype(str))


def add_vtk_darray(numpy_array, param, parent_xml_element):
    '''
        Create a subelement of DataArray type in parent_xml_element
        from a numpy_array. Uses xml.etree.ElementTree

        Parameters:
            numpy_array
            param: Dict with the key as the name of the DataArray
                attributes and item as the value of the latter.
            parent_xml_element: xml element.

    '''
    data_points_el = ET.SubElement(parent_xml_element, 'DataArray')
    for key, items in param.items():
        data_points_el.set(key, items)

    data_points_el.text = to_unicode(numpy_array)


def make_vtk_sub_element(sub_element_name, parent_element,
                         sub_array_att=None, data_array_data=None,
                         data_array_att=('type', 'Name', 'format'),
                         vtk_data_format='ascii'):
    '''
    Creates xml.etree.ElementTree subelement of
        parent_element of Cells type.

        Parameters:
            sub_element_name: string

            sub_array_att: dict() containing the attributes of the
                ET.SubElement()

            data_array_data: dict()
                The sub_element is a DataArray.
                Therefore the dict contains as keys the DataArray name
                and as item the DataArray np.array.

            data_array_att: tuple containing the attributes to be added
                in the DataArray subclass of this subelements.
                By default ('type', 'Name', 'format')

            parent_vtk_element: An instance of ET.Element()

            vtk_data_format: str
                By default 'ascii' format.

    '''

    if sub_array_att is None:
        sub_array_att = {}
    if data_array_data is None:
        data_array_data = {}

    # x = (key, item->np.array)
    functions_call_operator = {
            'type': lambda x: np_to_vtk_dtype(x[1].dtype.name),
            'Name': lambda x: x[0],
            'NumberOfComponents': lambda x: '{}'.format(x[1].shape[1])}

    sub_element = ET.SubElement(parent_element, sub_element_name,
                                sub_array_att)

    for key, item in data_array_data.items():

        x = (key, item)
        param = {key:functions_call_operator[key](x)
                 for key in data_array_att if key is not 'format'}
        param.update({'format':'{}'.format(vtk_data_format)})
        add_vtk_darray(
                numpy_array=item,
                param=param,
                parent_xml_element=sub_element)

    return sub_element


def make_UnstructuredGrid(cell_data, coordinates, points_data,
                          parent_vtk_element, vtk_data_format='ascii'):
    '''
    Creates xml.etree.ElementTree subelement of a
        VTKFile element of UnstructuredGrid type.
        For more information see [1]

        Parameters:
            cell_data: dict()
                Contains as keys: 'connectivity', 'offsets' and 'type'
                See Obs 1 for more information.

            coordinates: tuple or list of numpy arrays -> (x, y, z)
                or multidimensional numpy array

            points_data: dict() with key the name of the data and
                items a tuple containing :
                    1. numpy array containing the data
                        The length of
                        the latter must be the same as the number of
                        coordinates
                    2. the type of the data, i.e. Scalar,
                        Vector, etc.. See [1] for more

            vtk_data_format: str
                By default 'ascii' format.
                It is a heavy format.
                TODO: implement bytes with ... module.

            parent_vtk_element: An instance of ET.Element('VTKFile',
                                                          Attributes)

        Obs :
            Concerning the Cell structure in vtk:
                It has a cell offset : It is used to indicate the number of
                    points within a cell, i.e. it is the index of the last
                    node (point) in each cell.
                    More precisely :
                       It indexes to the end of the cell in the STL-iterator
                       sense where the end iterator is one past the last
                       element.  Therefore the value of the first offset
                       is the number of points in the first cell, and the
                       value of the last offset is the length of the
                       connectivity array. Taken from stackoverflow by ?

                It has a cell connectivity: In our case each point is connected
                    to itself.

                It has a cell type: In our case 1 (wich means point), see more
                    at [1].

        [1] : https://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf
    '''

    uGrid_element = ET.SubElement(parent_vtk_element, 'UnstructuredGrid')

    npoints = coordinates.shape[0]
    piece_element = ET.SubElement(uGrid_element, 'Piece',
                                  {'NumberOfPoints':'{}'.format(npoints),
                                   'NumberOfCells':'{}'.format(npoints)})
    # Make Points subelement
    make_vtk_sub_element(
            sub_element_name='Points',
            data_array_data={'Coordinates':coordinates},
            data_array_att=('type', 'NumberOfComponents', 'format'),
            parent_element=piece_element,
            vtk_data_format=vtk_data_format)

    # Make Cells subelement
    make_vtk_sub_element(
            sub_element_name='Cells',
            data_array_data=cell_data,
            parent_element=piece_element,
            vtk_data_format=vtk_data_format)

    # Make PointsData subelement
    make_vtk_sub_element(
            sub_element_name='PointData',
            sub_array_att={key:item[1] for key, item in points_data.items()},
            data_array_data={key:item[0] for key, item in points_data.items()},
            parent_element=piece_element,
            vtk_data_format=vtk_data_format)

    return uGrid_element


def points_vtk(filepath, coordinates, points_data,
               vtk_data_format='ascii', vkt_version='0.1',
               xml_declaration=False):
    '''
        From a set of coordinates and data points creates a xml file
        of UnsctructureGrid type (.vtu).

        Parameters:
            filepath: str

            coordinates: tuple or list of numpy arrays -> (x, y, z) or (x, y)
                or multidimensional numpy array

            points_data: dict() with key the name of the data and
                items a tuple containing :
                    1. numpy array containing the data
                        The length of
                        the latter must be the same as the number of
                        coordinates
                    2. the type of the data, i.e. Scalar,
                        Vector, etc.. See [1] for more

                e.g. : points_data={
                            'Temp': (temp, 'Scalar'),
                            'Pressure': (pressure, 'Scalar')},

            vtk_data_format: str
                By default 'ascii' format.
                It is a heavy format.
                TODO: implement bytes with ... module.

            vtk_version: str
                By default '0.1'. Works for ParaView 5.1.

        [1] :  https://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf
    '''

    if isinstance(coordinates, (tuple, list)):

        if len(coordinates) == 1:
            coordinates = np.vstack((
                    coordinates[0],
                    np.zeros(coordinates[0].size, dtype='int32'),
                    np.zeros(coordinates[0].size, dtype='int32'))).T

        elif len(coordinates) == 2:
            for pos, np_array in enumerate(coordinates[1:len(coordinates)]):
                if np_array.size != coordinates[
                        (pos - 1) % len(coordinates)].size:
                    raise ValueError(
                            'Input coordinates do have the same dimensions')

            coordinates = np.vstack((
                    coordinates[0],
                    coordinates[1],
                    np.zeros(coordinates[0].size, dtype='int32'))).T
        else:

            coordinates = np.vstack(coordinates).T
    else:

        if len(coordinates.shape) == 1:
            coordinates = coordinates[:, None]

        if coordinates.shape[1] == 1:
            coordinates = np.hstack((
                    coordinates,
                    np.zeros((coordinates.shape[0], 1), dtype='int32'),
                    np.zeros((coordinates.shape[0], 1), dtype='int32')))

        elif coordinates.shape[1] == 2:
            coordinates = np.hstack((
                coordinates,
                np.zeros((coordinates.shape[0], 1), dtype='int32')))

    npoints = coordinates.shape[0]

    # Check if the array is contingous
    assert (coordinates.flags['C_CONTIGUOUS'] or
            coordinates.flags['F_CONTIGUOUS'])


    cell_data = {'connectivity': np.arange(npoints, dtype='int32'),
                 'offsets': np.arange(1, npoints + 1, dtype='int32'),
                 'types': np.ones(npoints, dtype='uint8')}

    if vtk_data_format is 'ascii':

        if npoints > 1e6:
            raise warnings.WarningMessage('There are {0} points. \
                                          therefore the file will be large. \
                                          You can use the "binary" option \
                                          with PyEVTK\
                                          module installed (c.f pip \
                                          install pyevtk), or save it as \
                                          HDF5 or NetCDF')

        vtk_element = ET.Element('VTKFile',
                                 {'type':'UnstructuredGrid',
                                  'version':'{}'.format(vkt_version),
                                  'header_type':'UInt64'})

        make_UnstructuredGrid(
                cell_data=cell_data,
                coordinates=coordinates,
                points_data=points_data,
                parent_vtk_element=vtk_element,
                vtk_data_format=vtk_data_format)

        tree = ET.ElementTree(vtk_element)
        filepath = filepath + '.vtu'
        tree.write(filepath, xml_declaration=xml_declaration,
                   encoding=vtk_data_format)

    if vtk_data_format is 'binary':

        # Needs to be deepcopied so that is is contingous
        x = copy.deepcopy(coordinates[:, 0])
        y = copy.deepcopy(coordinates[:, 1])
        z = copy.deepcopy(coordinates[:, 2])

        try:
            pyevtk.pointsToVTK(
                    "./points", x, y, z,
                    data = {key:item[0] for key, item in points_data.items()})
        except NameError:
            import  pyevtk.hl as pyevtk

            pyevtk.pointsToVTK(
                    "./points", x, y, z,
                    data = {key:item[0] for key, item in points_data.items()})

# Generating random point data

#current = os.path.dirname(os.path.abspath(__file__))
#
#npoints = 100000
#np.random.seed(seed=9719819)
#x = np.random.rand(npoints)
#y = np.random.rand(npoints)
#z = np.random.rand(npoints)
#
#pressure = np.random.rand(npoints)
#temp = np.random.rand(npoints)
#
#points_vtk(filepath=current + 'slower',
#           coordinates=(x, y, z),
#           points_data={
#                   'Temp': (temp, 'Scalar'),
#                   'Pressure': (pressure, 'Scalar')},
#           vtk_data_format='ascii', vkt_version='0.1', xml_declaration=False)
