set(FreeImage_FOUND FALSE)

set(FreeImage_HINTS 
	/usr/local
	$ENV{FREEIMAGE_ROOT}
)

if(NOT FreeImage_ROOT_DIR)
    find_path(FreeImage_ROOT_DIR 
		NAMES include/FreeImage.h
		HINTS ${FreeImage_HINTS}
	)
endif()

find_path(FreeImage_INCLUDE_DIR 
	NAMES FreeImage.h
	HINTS ${FreeImage_ROOT_DIR}
	PATH_SUFFIXES include
)

find_library(FreeImage_LIBRARY_RELEASE 
	NAMES FreeImage
	HINTS ${FreeImage_ROOT_DIR}
	PATH_SUFFIXES lib
)

if(NOT FreeImage_LIBRARY_DEBUG)
    find_library(FreeImage_LIBRARY_DEBUG 
		NAMES FreeImaged
		HINTS ${FreeImage_ROOT_DIR}
		PATH_SUFFIXES lib
	)
endif()

if(FreeImage_LIBRARY_DEBUG)
	set(FreeImage_LIBRARY optimized ${FreeImage_LIBRARY_RELEASE} debug ${FreeImage_LIBRARY_DEBUG})
else()
	set(FreeImage_LIBRARY ${FreeImage_LIBRARY_RELEASE})
endif()

if (FreeImage_INCLUDE_DIR AND FreeImage_LIBRARY)
	set(FreeImage_FOUND TRUE)
	message(STATUS "Found FreeImage at ${FreeImage_ROOT_DIR}")
endif()
