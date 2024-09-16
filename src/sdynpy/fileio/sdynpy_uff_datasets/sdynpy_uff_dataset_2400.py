# -*- coding: utf-8 -*-
"""
Model Header

This dataset defines the header for a model in a universal file

Copyright 2022 National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import datetime
from ..sdynpy_uff import UFFReadError


class Sdynpy_UFF_Dataset_2400:
    def __init__(self,
                 model_uid=100,
                 entity_type=7,
                 entity_subtype=1,
                 version_number=-1,
                 entity_name='sdynpy_geometry',
                 part_number='',
                 status_mask=[False] * 32,
                 datetime_short_time_format=None,
                 idm_item_version_id=1,
                 idm_item_id=1,
                 primary_parent_uid=9,
                 geometry_switch=False,
                 p_analysis_switch=False,
                 all_selections_switch=False,
                 auto_create_dynamic_groups_switch=False,
                 acdg_1d_element_switch=False,
                 acdg_2d_element_switch=False,
                 acdg_3d_element_switch=False,
                 acdg_other_element_switch=False,
                 acdg_related_nodes_switch=False,
                 acdg_related_geometry_switch=False,
                 acdg_related_boundary_condition_switch=False):
        self.model_uid = model_uid
        self.entity_type = entity_type
        self.entity_subtype = entity_subtype
        self.version_number = version_number
        self.entity_name = entity_name
        self.part_number = part_number
        self.status_mask = status_mask
        self.datetime_short_time_format = datetime_short_time_format
        self.idm_item_version_id = idm_item_version_id
        self.idm_item_id = idm_item_id
        self.primary_parent_uid = primary_parent_uid
        self.geometry_switch = geometry_switch
        self.p_analysis_switch = p_analysis_switch
        self.all_selections_switch = all_selections_switch
        self.auto_create_dynamic_groups_switch = auto_create_dynamic_groups_switch
        self.acdg_1d_element_switch = acdg_1d_element_switch
        self.acdg_2d_element_switch = acdg_2d_element_switch
        self.acdg_3d_element_switch = acdg_3d_element_switch
        self.acdg_other_element_switch = acdg_other_element_switch
        self.acdg_related_nodes_switch = acdg_related_nodes_switch
        self.acdg_related_geometry_switch = acdg_related_geometry_switch
        self.acdg_related_boundary_condition_switch = acdg_related_boundary_condition_switch

    @property
    def dataset_number(self):
        return 2400

    @classmethod
    def from_uff_data_array(cls, data):
        # Transform from binary to ascii
        data = [line.decode() for line in data]
        # Record 1: FORMAT(I12,2I6,I12)
        #                 Field 1       -- Model UID
        #                 Field 2       -- Entity type
        #                 Field 3       -- Entity subtype
        #                 Field 4       -- Version number
        try:
            model_uid = int(data[0][0:12])
        except ValueError:
            model_uid = data[0][0:12].strip()
        try:
            entity_type = int(data[0][12:18])
        except ValueError:
            entity_type = data[0][12:18].strip()
        try:
            entity_subtype = int(data[0][18:24])
        except ValueError:
            entity_subtype = data[0][18:24].strip()
        try:
            version_number = int(data[0][24:36])
        except ValueError:
            version_number = data[0][24:36].strip()
        #
        # Record 2: FORMAT(40A2)
        #                 Field 1       -- Entity name
        entity_name = data[1][0:40].strip()
        #
        # Record 3: FORMAT(40A2)
        #                 Field 1       -- Part number
        part_number = data[2][0:40].strip()
        #
        # Record 4: FORMAT(32I2)
        #                 Field 1-32    -- Status mask
        try:
            status_mask = int(data[3][0:32])
        except ValueError:
            status_mask = data[3][0:32].strip()
        #
        # Record 5: FORMAT(10A2,3I12)
        #                 Field 1-2     -- Date/time short time format
        #                 Field 3       -- IDM item version ID
        #                 Field 4       -- IDM item ID
        #                 Field 5       -- Primary parent UID
        try:
            datetime_short_time_format = datetime.datetime.strptime(
                ''.join(data[4][0:20].split()), '%d-%b-%y%H:%M:%S')
        except ValueError:
            datetime_short_time_format = data[4][0:20].strip()
        try:
            idm_item_version_id = int(data[4][20:32])
        except ValueError:
            idm_item_version_id = data[4][20:32].strip()
        try:
            idm_item_id = int(data[4][32:44])
        except ValueError:
            idm_item_id = data[4][32:44].strip()
        try:
            primary_parent_uid = int(data[4][44:56])
        except ValueError:
            primary_parent_uid = data[4][44:56].strip()
        #
        # Record 6: FORMAT(I12)
        #                 Field 1       -- Optimization switch bitmask
        #                                  Bit 1 = Geometry switch
        #                                  Bit 2 = P analysis switch
        #                                  Bit 3 = All sections switch
        #                                  Bit 4 = Auto create dynamic groups switch
        #                                  Bit 5 = Auto create dynamic group -
        #                                          1D element switch
        #                                  Bit 6 = Auto create dynamic group -
        #                                          2D element switch
        #                                  Bit 7 = Auto create dynamic group -
        #                                          3D element switch
        #                                  Bit 8 = Auto create dynamic group -
        #                                          Other element switch
        #                                  Bit 9 = Auto create dynamic group -
        #                                          Related nodes switch
        #                                  Bit 10= Auto create dynamic group -
        #                                          Related geometry switch
        #                                  Bit 11= Auto create dynamic group -
        #                                          Related boundary condition switch
        try:
            integer = int(data[5][0:12])
            # TODO: I'm not sure if bit 1 is the most significant bit or least significant bit.
            # I will assume most significant
            geometry_switch = bool(integer & 2**10)
            p_analysis_switch = bool(integer & 2**9)
            all_selections_switch = bool(integer & 2**8)
            auto_create_dynamic_groups_switch = bool(integer & 2**7)
            acdg_1d_element_switch = bool(integer & 2**6)
            acdg_2d_element_switch = bool(integer & 2**5)
            acdg_3d_element_switch = bool(integer & 2**4)
            acdg_other_element_switch = bool(integer & 2**3)
            acdg_related_nodes_switch = bool(integer & 2**2)
            acdg_related_geometry_switch = bool(integer & 2**1)
            acdg_related_boundary_condition_switch = bool(integer & 2**0)
        except ValueError:
            geometry_switch = False
            p_analysis_switch = False
            all_selections_switch = False
            auto_create_dynamic_groups_switch = False
            acdg_1d_element_switch = False
            acdg_2d_element_switch = False
            acdg_3d_element_switch = False
            acdg_other_element_switch = False
            acdg_related_nodes_switch = False
            acdg_related_geometry_switch = False
            acdg_related_boundary_condition_switch = False

        ds_2400 = cls(model_uid, entity_type, entity_subtype, version_number,
                      entity_name, part_number, status_mask,
                      datetime_short_time_format, idm_item_version_id,
                      idm_item_id, primary_parent_uid, geometry_switch,
                      p_analysis_switch, all_selections_switch,
                      auto_create_dynamic_groups_switch,
                      acdg_1d_element_switch, acdg_2d_element_switch,
                      acdg_3d_element_switch, acdg_other_element_switch,
                      acdg_related_nodes_switch, acdg_related_geometry_switch,
                      acdg_related_boundary_condition_switch)
        return ds_2400

    def __repr__(self):
        return 'Sdynpy_UFF_Dataset_2400<{}>'.format(self.entity_name)

    def write_string(self):
        try:
            datetime_short_time_format = ('{:<10}'.format(self.datetime_short_time_format.strftime('%d-%b-%y')) +
                                          '{:<10}'.format(self.datetime_short_time_format.strftime('%H:%M:%S')))
        except AttributeError:
            datetime_short_time_format = self.datetime_short_time_format
        if isinstance(self.status_mask, str):
            status_mask = self.status_mask
        else:
            status_mask = ''.join(['{:>2}'.format(int(bool_val)) for bool_val in self.status_mask])
        # TODO: I'm not sure if this is the correct bit order
        optimization_switch_bitmask = sum([2**(10 - i) for i, bool_val in
                                           enumerate([self.geometry_switch,
                                                      self.p_analysis_switch,
                                                      self.all_selections_switch,
                                                      self.auto_create_dynamic_groups_switch,
                                                      self.acdg_1d_element_switch,
                                                      self.acdg_2d_element_switch,
                                                      self.acdg_3d_element_switch,
                                                      self.acdg_other_element_switch,
                                                      self.acdg_related_nodes_switch,
                                                      self.acdg_related_geometry_switch,
                                                      self.acdg_related_boundary_condition_switch]) if bool_val])

        return ('{:>12}{:>6}{:>6}{:>12}\n'.format(self.model_uid,
                                                  self.entity_type,
                                                  self.entity_subtype,
                                                  self.version_number) +
                '{:<40}\n'.format(self.entity_name) +
                '{:<40}\n'.format(self.part_number) +
                '{:<64}\n'.format(status_mask) +
                '{:<20}{:<12}{:<12}{:<12}\n'.format(datetime_short_time_format,
                                                    self.idm_item_version_id,
                                                    self.idm_item_id,
                                                    self.primary_parent_uid) +
                '{:>12}\n'.format(optimization_switch_bitmask))

    def __str__(self):
        lines = self.write_string().split('\n')
        if len(lines) > 8:
            return 'Dataset 2400: Model Header\n  ' + '\n  '.join(lines[0:5] + ['.', '.', '.'])
        else:
            return 'Dataset 2400: Model Header\n  ' + '\n  '.join(lines)


def read(data):
    return Sdynpy_UFF_Dataset_2400.from_uff_data_array(data)
