/*******************************************************************************
Copyright (c) 2010, Jonathan Hiller (Cornell University)
If used in publication cite "J. Hiller and H. Lipson "Dynamic Simulation of Soft Heterogeneous Objects" In press. (2011)"

This file is part of Voxelyze.
Voxelyze is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Voxelyze is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
See <http://www.opensource.org/licenses/lgpl-3.0.html> for license details.
*******************************************************************************/

#include "VX_SimGA.h"
#include <iostream>

CVX_SimGA::CVX_SimGA()
{
	Fitness = 0.0f;
	TrackVoxel = 0;
	FitnessFileName = "";
//	print_scrn = false;
	WriteFitnessFile = false;
	FitnessType = FT_NONE;	//no reporting is default

}

void CVX_SimGA::SaveResultFile(std::string filename)
{
	CXML_Rip XML;
	WriteResultFile(&XML);
	XML.SaveFile(filename);
}


void CVX_SimGA::WriteResultFile(CXML_Rip* pXML)
{
// 	float totalPoints = 0;
// 	float goodSteps = 0;
// 	float badSteps = 0;
// 	for (std::map< float, float >::iterator it = floorIsLava.begin(); it != floorIsLava.end(); it++ )
// 	{
// 		totalPoints += it->second;
// 		if (it->second > 0 ) {goodSteps++;}
// 		if (it->second < 0 ) {badSteps++;}
// 	}
// 	// std::cout << "totalPoints: " << totalPoints << std::endl;
	float dist = pow(pow(SS.CurCM.x-IniCM.x,2)+pow(SS.CurCM.y-IniCM.y,2),0.5);	

	pXML->DownLevel("Voxelyze_Sim_Result");
	pXML->SetElAttribute("Version", "1.0");
	pXML->DownLevel("Fitness");
	// pXML->Element("FinalCOM_Dist", SS.CurCM.Normalize()-IniCM.Normalize());
	// pXML->Element("FinalCOM_DistX", SS.CurCM.x-IniCM.x);
	// pXML->Element("FinalCOM_DistY", SS.CurCM.y-IniCM.y);
	// pXML->Element("FinalCOM_DistZ", SS.CurCM.z-IniCM.z);
	// pXML->Element("DistanceFitness", SS.CurCM.Normalize()-IniCM.Normalize());
	// pXML->Element("EnergyFitness", 0);
	// pXML->Element("StepsIn", stepsIn);
	// pXML->Element("StepsOut", stepsOut);
	// pXML->Element("GoodSteps", goodSteps);
	// pXML->Element("BadSteps", badSteps);
	// pXML->Element("DistanceFitness", totalPoints);
	// pXML->Element("CompositeFitness", (SS.CurCM-IniCM).Length()/LocalVXC.GetLatticeDim()*(COMZ/numSamples) );
	// pXML->Element("Height", COMZ/numSamples );
	pXML->Element("CompositeFitness", dist/LocalVXC.GetLatticeDim()*COMZ );
	pXML->Element("Height", COMZ );
	pXML->Element("Distance", dist/LocalVXC.GetLatticeDim() );
	pXML->UpLevel();
	pXML->UpLevel();

	// std::cout << "dist: " << dist/LocalVXC.GetLatticeDim()  << std::endl;
	// // std::cout << "height: " << COMZ << std::endl;
	// std::cout << "fitness: " << dist/LocalVXC.GetLatticeDim() * COMZ << std::endl;

}

void CVX_SimGA::WriteAdditionalSimXML(CXML_Rip* pXML)
{
	pXML->DownLevel("GA");
		pXML->Element("Fitness", Fitness);
		pXML->Element("FitnessType", (int)FitnessType);
		pXML->Element("TrackVoxel", TrackVoxel);
		pXML->Element("FitnessFileName", FitnessFileName);
		pXML->Element("WriteFitnessFile", WriteFitnessFile);
	pXML->UpLevel();
}

bool CVX_SimGA::ReadAdditionalSimXML(CXML_Rip* pXML, std::string* RetMessage)
{
	if (pXML->FindElement("GA")){
		int TmpInt;
		if (!pXML->FindLoadElement("Fitness", &Fitness)) Fitness = 0;
		if (pXML->FindLoadElement("FitnessType", &TmpInt)) FitnessType=(FitnessTypes)TmpInt; else Fitness = 0;
		if (!pXML->FindLoadElement("TrackVoxel", &TrackVoxel)) TrackVoxel = 0;
		if (!pXML->FindLoadElement("FitnessFileName", &FitnessFileName)) FitnessFileName = "";
		if (!pXML->FindLoadElement("WriteFitnessFile", &WriteFitnessFile)) WriteFitnessFile = true;
		pXML->UpLevel();
	}

	return true;
}