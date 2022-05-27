cmake_minimum_required(VERSION 3.5)

project(betel_display LANGUAGES CXX)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(INC_DIR ./include)        #设置头文件路径变量
set(LINK_DIR ./depends)     #设置库路径变量
include_directories(${INC_DIR})    #包含第三方头文件
link_directories(${LINK_DIR})        #链接第三方库   与头文件是匹配的

set(Torch_DIR ./torch/share/cmake/Torch)   #Torch没有在指定路径下提供类似于opencv通过cmake提供的xxxconfig.cmake的文件，因此需要设置这样一个变量指向这个文件
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)

find_package(jsoncpp CONFIG REQUIRED)
INCLUDE_DIRECTORIES(${JSON_INC_PATH})

find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(Qt5 COMPONENTS SerialPort REQUIRED)

add_executable(betel_display
    main.cpp
    widget.cpp
    betelthread.cpp
    widget.h
    betelthread.h
    MessageQue.h
    widget.ui
)

target_link_libraries(betel_display PRIVATE Qt5::Widgets Qt5::SerialPort ImageConvert MVSDK VideoRender jsoncpp_lib ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET betel_display PROPERTY CXX_STANDARD 14)
