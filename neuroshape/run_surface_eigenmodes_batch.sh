# Function: get_batch_options
# Description
#
#   Retrieve the following command line parameter values if specified
#
#   --StudyFolder= - primary study folder containing subject ID subdirectories
#   --Subjlist=    - quoted, space separated list of subject IDs on which
#                    to run the pipeline
#
#   Set the values of the following global variables to reflect command
#   line specified parameters
#
#   command_line_specified_study_folder
#   command_line_specified_subj_list
#   command_line_specified_run_local
#
#   These values are intended to be used to override any values set
#   directly within this script file
get_batch_options() {
	local arguments=("$@")

	unset command_line_specified_study_folder
	unset command_line_specified_subj
	unset command_line_specified_structure
	unset command_line_specified_num_modes
	unset command_line_specified_normalization_type
	unset command_line_specified_normalization_factor

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
                command_line_specified_structure=${argument#*=}
                index=$(( index + 1 ))
                ;;
            --n_modes=*)
                command_line_specified_num_modes=${argument#*=}
                index=$(( index + 1 ))
                ;;
            --norm=*)
                command_line_specified_normalization_type=${argument#*=}
                index=$(( index + 1 ))
                ;;
            --norm_factor=*)
                command_line_specified_normalization_factor=${argument#*=}
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

    structure="striatum"
    n_modes=31
    norm="none"
    norm_factor=1
    hemispheres="lh rh"
    
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
		structure="${command_line_specified_structure}"
	fi
	
	if [ -n "${command_line_specified_num_modes}" ]; then
		n_modes="${command_line_specified_num_modes}"
	fi
	
	if [ -n "${command_line_specified_normalization_type}" ]; then
		norm="${command_line_specified_normalization_type}"
	fi
	
	if [ -n "${command_line_specified_normalization_factor}" ]; then
		norm="${command_line_specified_normalization_factor}"
	fi

	# Report major script control variables to user
	echo "StudyFolder: ${StudyFolder}"
	echo "FreeSurfer Folder: ${fs_subjects_dir}"
	echo "Subjlist: ${Subjlist}"
	echo "Structure: ${structure}"
	echo "Number of modes: ${n_modes}"
	echo "Normalization type: ${norm}"
	echo "Normalization factor: ${norm_factor}"
	
	# Check which structure to extract and calculate eigenmodes on
	
	###########################################################
    # Common labels to extract:                               #
    # Left_Caudate            : 11                            #
    # Left_Putamen            : 12                            #
    # Left_Accumbens_Area     : 26                            #
    # Right_Caudate           : 50                            #
    # Right_Putamen           : 51                            #
    # Right_Accumbens_Area    : 58                            #
    #                                                         #
    # Can also extract special labels and combine them:       #
    # Striatum                : Left_Caudate, Left_Putamen,   #
    #                           Left_Accumbens_Area,          #
    #                           Right_Caudate, Right_Putamen, #
    #                           Right_Accumbens_Area          #
    #                                                         #
    ###########################################################
    # TODO: IMPLEMENT OTHERS
    
    
    #local special_structures=("striatum")
    #local structure="$1"
    #if [ "$special_structure" =~ "$structure" ] ; then
    #    if [ $structure = "striatum" ]; then
    #    
    #        labels="11 12 26 50 51 58"       
    #        
    #    fi
    #fi        
    
    labels_lh=("11 12 26")
    labels_rh=("50 51 58")
	# Cycle through specified subjects
	for Subject in $Subjlist ; do
		echo $Subject

		# Input Images

		# Detect aseg.mgz images and build list of full paths
		numsegws=`ls ${fs_subjects_dir}/${Subject}/mri/aseg.mgz | wc -l`
		echo "Found ${numsegws} FreeSurfer Segmentation Images for subject ${Subject}"
		if [ $numsegws -gt 1 ]; then
    		echo ""
            echo "ERROR: Too many segmentation images: ${argument}"
            echo ""
            exit 1
        fi
		segInputImage="${fs_subjects_dir}/${Subject}/mri/aseg.mgz"
		segConverted="${StudyFolder}/${Subject}_aseg.nii.gz"
		echo "Converting Input Image : ${segInputImage} to ${segConverted}"
		mrconvert ${segInputImage} ${segConverted} -quiet -force

		# Extract label images for volume eigenmode calculation
		for label in $labels_lh ; do
    		echo "Extracting label ${label} from FreeSurfer segmentation image ${segInputImages}"
    		outputImage="${StudyFolder}/${Subject}_${label}_lh.nii.gz"
        	echo "Output Image : ${outputImage}"
        	fslmaths ${segConverted} -thr ${label} -uthr ${label} -bin ${outputImage}
        done
        for label in $labels_rh ; do
            echo "Extracting label ${label} from FreeSurfer segmentation image ${segInputImages}"
            outputImage="${StudyFolder}/${Subject}_${label}_rh.nii.gz"
            echo "Output Image : ${outputImage}"
            fslmaths ${segConverted} -thr ${label} -uthr ${label} -bin ${outputImage}
        done
        
        # Now find the label images just made and combine them
        echo "Combining label images into ${structure}"
        
        for hemisphere in ${hemispheres}; do
            echo Processing ${hemisphere}
            
            if [ ${hemisphere} == 'lh' ]; then
                labels=$labels_lh
            else
                labels=$labels_rh
            fi
                
            first=${StudyFolder}/${Subject}_${labels:0:2}_${hemisphere}.nii.gz 
            seg=""
            B=("${labels:2}")    
            for label in $B; do
                seg+=" ${StudyFolder}/${Subject}_${label}_${hemisphere}.nii.gz"
            done
                       
            command_list="${first}"
            
            for image in $seg ; do
                command_list="${command_list} -add ${image}"
            done
            
            outputName=${StudyFolder}/${Subject}_${structure}_${hemisphere}
            echo "Command: fslmaths ${command_list} -bin ${outputName}"
            fslmaths ${command_list} -bin ${outputName}
            
            echo "Dilating mask so components stay connected"
            echo "Command: fslmaths ${outputName} -dilM ${outputName}_dilate"
            
            fslmaths ${outputName} -dilM ${outputName}_dilate
            mv ${outputName}_dilate.nii.gz ${outputName}.nii.gz
            
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