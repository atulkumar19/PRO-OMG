// COPYRIGHT 2015-2019 LEOPOLDO CARBAJAL

/*	This file is part of PROMETHEUS++.

    PROMETHEUS++ is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    PROMETHEUS++ is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PROMETHEUS++.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "structures.h"

double F_EPSILON_DS = F_EPSILON;    // Dimensionless vacuum permittivity
double F_E_DS = F_E;                // Dimensionless electric charge
double F_ME_DS = F_ME;              // Dimensionless electron mass
double F_MU_DS = F_MU;              // Dimensionless vacuum permeability
double F_C_DS = F_C;                // Dimensionless speed of light


simulationParameters::simulationParameters()
{
    oneDimensional::ionSpecies IONS_1D;
    typesInfo.ionSpecies_1D_type = &typeid(IONS_1D);

    twoDimensional::ionSpecies IONS_2D;
    typesInfo.ionSpecies_2D_type = &typeid(IONS_2D);

    oneDimensional::fields EB_1D;
    typesInfo.fields_1D_type = &typeid(EB_1D);

    twoDimensional::fields EB_2D;
    typesInfo.fields_2D_type = &typeid(EB_2D);

    currentTime = 0;

    //std::cout << "1-D Ions Type: " << typesInfo.ionSpecies_1D_type->name() << std::endl;
}
