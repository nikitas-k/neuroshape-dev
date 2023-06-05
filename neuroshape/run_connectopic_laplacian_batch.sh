# Function: get_batch_options
# Description
#
#   Retrieve the following command line parameter values if specified
#
#   --StudyFolder= - primary study folder containing subject ID subdirectories
#                    VERY IMPORTANT NOTE!: THIS MUST BE AN ABSOLUTE PATH I.E.,
#                    NOT QUOTED RELATIVE FROM THE CURRENT WORKING DIRECTORY, OTHERWISE
#                    THIS CODE WILL NOT WORK!
#   --Subjlist=    - quoted, space separated list of subject IDs on which
#                    to run the pipeline
#   --Structures=  - quoted, space separated list of ROI images on which to run
#                    to run the pipeline. IMPORTANT: all ROIs must be in MNI152 1mm or 2mm
#                    space, otherwise the registration step will fail. All ROIs must
#                    also be binary label images - i.e., have voxel value of 1 where ROI is
#                    and 0 elsewhere. They must also not have an extension when given 
#                    to the command, but have .nii.gz or .nii extension otherwise. 
#                    This script can also handle a single input 
#                    ROI via either quoted:
#                    --Structures="roi"
#                    or unquoted
#                    --structure=roi   
#
#   Set the values of the following global variables to reflect command
#   line specified parameters
#
#   command_line_specified_study_folder
#   command_line_specified_subj_list
#   command_line_specified_structures
#   command_line_specified_num_gradients
#
#   These values are intended to be used to override any values set
#   directly within this script file
get_batch_options() {
	local arguments=("$@")

	unset command_line_specified_study_folder
	unset command_line_specified_subj
	unset command_line_specified_structures
	unset command_line_specified_num_gradients

	local index=0
	local numArgs=${#arguments[@]}
	local argument

	while [ ${index} -lt ${numArgs} ]; do
		argument=${arguments[index]}

		case ${argument} in
			--StudyFolder=*)
				command_line_specified_study_folder=${argument#*=}
				index=$(( index + 1 ))
				;;
            --fs_subjects_dir=*)
                command_line_specified_FreeSurfer_folder=${argument#*=}
                index=$(( index + 1 ))
                ;;
			--Subject=*)
				command_line_specified_subj=${argument#*=}
				index=$(( index + 1 ))
				;;
            --structure=*)
                command_line_specified_structures+=${argument#*=}
                index=$(( index + 1 ))
                ;;
            --n_gradients=*)
                command_line_specified_num_gradients=${argument#*=}
                index=$(( index + 1 ))
                ;;
            			
			*)
				echo ""
				echo "ERROR: Unrecognized Option: ${argument}"
				echo ""
				exit 1
				;;
		esac
	done
}

main()
{
	get_batch_options "$@"

	# Set variable values that locate and specify data to process
	StudyFolder="/Volumes/Scratch/functional_integration_psychosis/preprocessed/HCP-EP/LBO/striatum" # Location of Subject folders (named by subjectID)
	fs_subjects_dir="/Volumes/Scratch/functional_integration_psychosis/preprocessing/HCP-EP/FS"
	Subjlist="sub-1001 sub-1002 sub-1003 sub-1005 sub-1006 sub-1009 \
	sub-1010 sub-1012 sub-1015 sub-1017 sub-1018 sub-1019 sub-1020 \
	sub-1021 sub-1022 sub-1024 sub-1025 sub-1026 sub-1027 sub-1028 \
	sub-1029 sub-1030 sub-1031 sub-1032 sub-1033 sub-1034 sub-1035 \
	sub-1036 sub-1037 sub-1038 sub-1039 sub-1040 sub-1041 sub-1043 \
	sub-1044 sub-1045 sub-1047 sub-1048 sub-1050 sub-1051 sub-1052 \
	sub-1053 sub-1054 sub-1056 sub-1057 sub-1060 sub-1061 sub-1063 \
	sub-1064 sub-1065 sub-1066 sub-1067 sub-1068 sub-1070 sub-1071 \
	sub-1072 sub-1073 sub-1074 sub-1075 sub-1076 sub-1077 sub-1078 \
	sub-1079 sub-1080 sub-1081 sub-1082 sub-1083 sub-1084 sub-1085 \
	sub-1086 sub-1087 sub-1088 sub-1089 sub-1091 sub-1093 sub-1094 \
	sub-1095 sub-1098 sub-1099 sub-1104 sub-1105 sub-2004 sub-2005 \
	sub-2006 sub-2007 sub-2008 sub-2010 sub-2012 sub-2014 sub-2015 \
	sub-2016 sub-2019 sub-2020 sub-2022 sub-2023 sub-2029 sub-2031 \
	sub-2033 sub-2038 sub-2040 sub-2041 sub-2042 sub-2044 sub-2045 \
	sub-2049 sub-2052 sub-2062 sub-2065 sub-3002 sub-3009 sub-3011 \
	sub-3017 sub-3020 sub-3021 sub-3022 sub-3025 sub-3026 sub-3027 \
	sub-3028 sub-3029 sub-3031 sub-3032 sub-3034 sub-3035 sub-3039 \
	sub-4002 sub-4003 sub-4004 sub-4005 sub-4006 sub-4010 sub-4011 \
	sub-4012 sub-4014 sub-4015 sub-4018 sub-4022 sub-4023 sub-4024 \
	sub-4027 sub-4028 sub-4029 sub-4030 sub-4031 sub-4035 sub-4036 \
	sub-4037 sub-4038 sub-4040 sub-4047 sub-4048 sub-4049 sub-4050 \
	sub-4052 sub-4053 sub-4057 sub-4058 sub-4059 sub-4063 sub-4064 \
	sub-4065 sub-4069 sub-4071 sub-4072 sub-4075 sub-4088 sub-4091"                                # Space delimited list of subject IDs

    Structures="masks/striatum_2mm"
    n_gradients=31
    
	# Use any command line specified options to override any of the variable settings above
	if [ -n "${command_line_specified_study_folder}" ]; then
		StudyFolder="${command_line_specified_study_folder}"
	fi
	
	if [ -n "${command_line_specified_FreeSurfer_folder}" ]; then
		fs_subjects_dir="${command_line_specified_FreeSurfer_folder}"
	fi

	if [ -n "${command_line_specified_subj}" ]; then
		Subjlist="${command_line_specified_subj}"
	fi
	
	if [ -n "${command_line_specified_structure}" ]; then
		Structures="${command_line_specified_structure}"
	fi
	
	if [ -n "${command_line_specified_num_modes}" ]; then
		n_modes="${command_line_specified_num_gradients}"
	fi

	# Report major script control variables to user
	echo "StudyFolder: ${StudyFolder}"
	echo "FreeSurfer Folder: ${fs_subjects_dir}"
	echo "Subjlist: ${Subjlist}"
	echo "Structures: ${Structures}"
	echo "Number of modes: ${n_gradients}"
	
	SUBJECTS_DIR=${fs_subjects_dir}
	export $SUBJECTS_DIR
	# mask folder
	MASK=$(pwd)/masks/GMmask.nii
	# Cycle through specified subjects
	for Subject in $Subjlist ; do
		echo $Subject

		# Set up folder structure
		mkdir -p ${StudyFolder}/${Subject}
		cd ${StudyFolder}/${Subject}

		# Detect aseg.mgz images and build list of full paths
		numsegws=`ls ${fs_subjects_dir}/${Subject}/mri/aseg.mgz | wc -l`
		echo "Found ${numsegws} FreeSurfer Segmentation Images for subject ${Subject}"
		if [ $numsegws -gt 1 ]; then
    		echo ""
            echo "ERROR: Too many segmentation images: ${argument}"
            echo ""
            exit 1
        fi
        
        # Detect fmri images and build list of full paths
        
        
		segInputImage="${fs_subjects_dir}/${Subject}/mri/aseg.mgz"
		segConverted="aseg_mni.nii.gz"
		echo "Converting Input Image to Subject space : ${segInputImage} to ${segConverted}"
		tkregister2 --mov ${SUBJECTS_DIR}/${Subject}/mri/brain.mgz --targ ${SUBJECTS_DIR}/${Subject}/mri/rawavg.mgz --reg register.native.dat --noedit --regheader --fslregout FS2FSL.mat
		mri_vol2vol --mov ${SUBJECTS_DIR}/${Subject}/mri/brain.mgz --targ ${SUBJECTS_DIR}/${Subject}/mri/rawavg.mgz --regheader -o brainFSnat.nii
		
		mri_vol2vol --mov ${segInputImage} --targ ${SUBJECTS_DIR}/${Subject}/mri/rawavg.mgz --regheader -o ${segConverted}
	    labelconvert ${segConverted} $FREESURFER_HOME/FreeSurferColorLUT.txt $MRTRIX/share/mrtrix3/labelconvert/fs_default.txt parc.nii -nthreads 1 -force -quiet
	    mrconvert ${SUBJECTS_DIR}/${Subject}/mri/brain.mgz brainFS.nii -quiet -nthreads 1
	    
	    echo "Making GMmask.nii from segmentation image ${segConverted}"
	    fslmaths parc.nii -bin GMmask.nii.gz
	    gunzip -f GMmask.nii.gz
        
		# Do connectopic Laplacian
		#cycle through structures
		for structure in ${Structures} ; do
    		echo "Registering ${label} to subject space"
    		vox_size=`mrinfo ${structure} | grep Vox | cut -d " " -f12`
    		dims=`mrinfo ${structure} | grep Dim | cut -d " " -f11`
    		dims+=" `mrinfo ${structure} | grep Dim | cut -d " " -f13`"
    		dims+=" `mrinfo ${structure} | grep Dim | cut -d " " -f15`"
    		
    		if [[ ${vox_size} == '2' ]]; then
        		standard=${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz
        		if [[ ${dims} == '91 109 91' ]]; then
            		flirt -ref brainFS.nii -in ${FSLDIR}/data/standard/MNI152_T1_2mm_brain -dof 12 -cost normmi -omat MNI2FS.mat
            	else
                	echo ""
    				echo "ERROR: ${structure} Must have the same dimensions as ${standard}, exiting."
    				echo ""
    				exit 1
        	elif [[ ${vox_size} == '1' ]]; then
            	standard=${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz
            	if [[ ${dims} == '182 218 182' ]]; then
                	flirt -ref brainFS.nii -in ${FSLDIR}/data/standard/MNI152_T1_1mm_brain -dof 12 -cost normmi -omat MNI2FS.mat
                else
                    echo ""
    				echo "ERROR: ${structure} Must have the same dimensions as ${standard}, exiting."
    				echo ""
    				exit 1
            else
                echo ""
				echo "ERROR: Voxel size must be 2mm or 1mm isotropic: ${structure}, exiting."
				echo ""
				exit 1
            fi
        	flirt -ref brainFS.nii -in ${structure} -applyxfm -init MNI2FS.mat -out ${structure}_FS
    		flirt -ref brainFSnat.nii -in ${structure}_FS -applyxfm -init FS2FSL.mat -out ${structure}_nat
    	
            nifti_input_filename=${outputName}.nii.gz
            nifti_output_filename=${outputName}_emode_31.nii.gz
            output_eval_filename=${outputName}_eval_31.txt
            output_emode_filename=${outputName}_emode_31.txt
            
            python volume_eigenmodes.py ${nifti_input_filename} ${nifti_output_filename} \
                                        ${output_eval_filename} ${output_emode_filename} \
                                        -N ${n_modes} -norm ${norm} \
                                        -normfactor ${norm_factor}
            
        done
        
        # Clean up temporary label files
        echo "Cleaning up temporary files..."
        forcleanup=`find -name "${Subject}_[0-9]*.nii.gz"`
        forcleanup+=" ${segConverted}"
        rm -f ${forcleanup}
    done
}

# Invoke the main function to get things started
main "$@"