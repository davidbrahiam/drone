// Generated by gencpp from file ardrone_autonomy/navdata_magneto.msg
// DO NOT EDIT!


#ifndef ARDRONE_AUTONOMY_MESSAGE_NAVDATA_MAGNETO_H
#define ARDRONE_AUTONOMY_MESSAGE_NAVDATA_MAGNETO_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>
#include <ardrone_autonomy/vector31.h>
#include <ardrone_autonomy/vector31.h>
#include <ardrone_autonomy/vector31.h>

namespace ardrone_autonomy
{
template <class ContainerAllocator>
struct navdata_magneto_
{
  typedef navdata_magneto_<ContainerAllocator> Type;

  navdata_magneto_()
    : header()
    , drone_time(0.0)
    , tag(0)
    , size(0)
    , mx(0)
    , my(0)
    , mz(0)
    , magneto_raw()
    , magneto_rectified()
    , magneto_offset()
    , heading_unwrapped(0.0)
    , heading_gyro_unwrapped(0.0)
    , heading_fusion_unwrapped(0.0)
    , magneto_calibration_ok(0)
    , magneto_state(0)
    , magneto_radius(0.0)
    , error_mean(0.0)
    , error_var(0.0)  {
    }
  navdata_magneto_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , drone_time(0.0)
    , tag(0)
    , size(0)
    , mx(0)
    , my(0)
    , mz(0)
    , magneto_raw(_alloc)
    , magneto_rectified(_alloc)
    , magneto_offset(_alloc)
    , heading_unwrapped(0.0)
    , heading_gyro_unwrapped(0.0)
    , heading_fusion_unwrapped(0.0)
    , magneto_calibration_ok(0)
    , magneto_state(0)
    , magneto_radius(0.0)
    , error_mean(0.0)
    , error_var(0.0)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef double _drone_time_type;
  _drone_time_type drone_time;

   typedef uint16_t _tag_type;
  _tag_type tag;

   typedef uint16_t _size_type;
  _size_type size;

   typedef int16_t _mx_type;
  _mx_type mx;

   typedef int16_t _my_type;
  _my_type my;

   typedef int16_t _mz_type;
  _mz_type mz;

   typedef  ::ardrone_autonomy::vector31_<ContainerAllocator>  _magneto_raw_type;
  _magneto_raw_type magneto_raw;

   typedef  ::ardrone_autonomy::vector31_<ContainerAllocator>  _magneto_rectified_type;
  _magneto_rectified_type magneto_rectified;

   typedef  ::ardrone_autonomy::vector31_<ContainerAllocator>  _magneto_offset_type;
  _magneto_offset_type magneto_offset;

   typedef float _heading_unwrapped_type;
  _heading_unwrapped_type heading_unwrapped;

   typedef float _heading_gyro_unwrapped_type;
  _heading_gyro_unwrapped_type heading_gyro_unwrapped;

   typedef float _heading_fusion_unwrapped_type;
  _heading_fusion_unwrapped_type heading_fusion_unwrapped;

   typedef uint8_t _magneto_calibration_ok_type;
  _magneto_calibration_ok_type magneto_calibration_ok;

   typedef uint32_t _magneto_state_type;
  _magneto_state_type magneto_state;

   typedef float _magneto_radius_type;
  _magneto_radius_type magneto_radius;

   typedef float _error_mean_type;
  _error_mean_type error_mean;

   typedef float _error_var_type;
  _error_var_type error_var;




  typedef boost::shared_ptr< ::ardrone_autonomy::navdata_magneto_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::ardrone_autonomy::navdata_magneto_<ContainerAllocator> const> ConstPtr;

}; // struct navdata_magneto_

typedef ::ardrone_autonomy::navdata_magneto_<std::allocator<void> > navdata_magneto;

typedef boost::shared_ptr< ::ardrone_autonomy::navdata_magneto > navdata_magnetoPtr;
typedef boost::shared_ptr< ::ardrone_autonomy::navdata_magneto const> navdata_magnetoConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::ardrone_autonomy::navdata_magneto_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::ardrone_autonomy::navdata_magneto_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace ardrone_autonomy

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': True}
// {'std_msgs': ['/opt/ros/indigo/share/std_msgs/cmake/../msg'], 'ardrone_autonomy': ['/home/davidbrahiam/tum_simulator_ws/src/ros_ardrone/ardrone_autonomy/msg'], 'geometry_msgs': ['/opt/ros/indigo/share/geometry_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::ardrone_autonomy::navdata_magneto_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::ardrone_autonomy::navdata_magneto_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::ardrone_autonomy::navdata_magneto_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::ardrone_autonomy::navdata_magneto_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::ardrone_autonomy::navdata_magneto_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::ardrone_autonomy::navdata_magneto_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::ardrone_autonomy::navdata_magneto_<ContainerAllocator> >
{
  static const char* value()
  {
    return "c256b1c1d1ff0cb12a13c36720b3e224";
  }

  static const char* value(const ::ardrone_autonomy::navdata_magneto_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xc256b1c1d1ff0cb1ULL;
  static const uint64_t static_value2 = 0x2a13c36720b3e224ULL;
};

template<class ContainerAllocator>
struct DataType< ::ardrone_autonomy::navdata_magneto_<ContainerAllocator> >
{
  static const char* value()
  {
    return "ardrone_autonomy/navdata_magneto";
  }

  static const char* value(const ::ardrone_autonomy::navdata_magneto_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::ardrone_autonomy::navdata_magneto_<ContainerAllocator> >
{
  static const char* value()
  {
    return "Header header\n\
float64 drone_time\n\
uint16 tag\n\
uint16 size\n\
int16 mx\n\
int16 my\n\
int16 mz\n\
vector31 magneto_raw\n\
vector31 magneto_rectified\n\
vector31 magneto_offset\n\
float32 heading_unwrapped\n\
float32 heading_gyro_unwrapped\n\
float32 heading_fusion_unwrapped\n\
uint8 magneto_calibration_ok\n\
uint32 magneto_state\n\
float32 magneto_radius\n\
float32 error_mean\n\
float32 error_var\n\
\n\
================================================================================\n\
MSG: std_msgs/Header\n\
# Standard metadata for higher-level stamped data types.\n\
# This is generally used to communicate timestamped data \n\
# in a particular coordinate frame.\n\
# \n\
# sequence ID: consecutively increasing ID \n\
uint32 seq\n\
#Two-integer timestamp that is expressed as:\n\
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n\
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n\
# time-handling sugar is provided by the client library\n\
time stamp\n\
#Frame this data is associated with\n\
# 0: no frame\n\
# 1: global frame\n\
string frame_id\n\
\n\
================================================================================\n\
MSG: ardrone_autonomy/vector31\n\
float32 x\n\
float32 y\n\
float32 z\n\
";
  }

  static const char* value(const ::ardrone_autonomy::navdata_magneto_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::ardrone_autonomy::navdata_magneto_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.drone_time);
      stream.next(m.tag);
      stream.next(m.size);
      stream.next(m.mx);
      stream.next(m.my);
      stream.next(m.mz);
      stream.next(m.magneto_raw);
      stream.next(m.magneto_rectified);
      stream.next(m.magneto_offset);
      stream.next(m.heading_unwrapped);
      stream.next(m.heading_gyro_unwrapped);
      stream.next(m.heading_fusion_unwrapped);
      stream.next(m.magneto_calibration_ok);
      stream.next(m.magneto_state);
      stream.next(m.magneto_radius);
      stream.next(m.error_mean);
      stream.next(m.error_var);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct navdata_magneto_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::ardrone_autonomy::navdata_magneto_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::ardrone_autonomy::navdata_magneto_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "drone_time: ";
    Printer<double>::stream(s, indent + "  ", v.drone_time);
    s << indent << "tag: ";
    Printer<uint16_t>::stream(s, indent + "  ", v.tag);
    s << indent << "size: ";
    Printer<uint16_t>::stream(s, indent + "  ", v.size);
    s << indent << "mx: ";
    Printer<int16_t>::stream(s, indent + "  ", v.mx);
    s << indent << "my: ";
    Printer<int16_t>::stream(s, indent + "  ", v.my);
    s << indent << "mz: ";
    Printer<int16_t>::stream(s, indent + "  ", v.mz);
    s << indent << "magneto_raw: ";
    s << std::endl;
    Printer< ::ardrone_autonomy::vector31_<ContainerAllocator> >::stream(s, indent + "  ", v.magneto_raw);
    s << indent << "magneto_rectified: ";
    s << std::endl;
    Printer< ::ardrone_autonomy::vector31_<ContainerAllocator> >::stream(s, indent + "  ", v.magneto_rectified);
    s << indent << "magneto_offset: ";
    s << std::endl;
    Printer< ::ardrone_autonomy::vector31_<ContainerAllocator> >::stream(s, indent + "  ", v.magneto_offset);
    s << indent << "heading_unwrapped: ";
    Printer<float>::stream(s, indent + "  ", v.heading_unwrapped);
    s << indent << "heading_gyro_unwrapped: ";
    Printer<float>::stream(s, indent + "  ", v.heading_gyro_unwrapped);
    s << indent << "heading_fusion_unwrapped: ";
    Printer<float>::stream(s, indent + "  ", v.heading_fusion_unwrapped);
    s << indent << "magneto_calibration_ok: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.magneto_calibration_ok);
    s << indent << "magneto_state: ";
    Printer<uint32_t>::stream(s, indent + "  ", v.magneto_state);
    s << indent << "magneto_radius: ";
    Printer<float>::stream(s, indent + "  ", v.magneto_radius);
    s << indent << "error_mean: ";
    Printer<float>::stream(s, indent + "  ", v.error_mean);
    s << indent << "error_var: ";
    Printer<float>::stream(s, indent + "  ", v.error_var);
  }
};

} // namespace message_operations
} // namespace ros

#endif // ARDRONE_AUTONOMY_MESSAGE_NAVDATA_MAGNETO_H
