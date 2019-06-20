// Generated by gencpp from file ardrone_autonomy/navdata_altitude.msg
// DO NOT EDIT!


#ifndef ARDRONE_AUTONOMY_MESSAGE_NAVDATA_ALTITUDE_H
#define ARDRONE_AUTONOMY_MESSAGE_NAVDATA_ALTITUDE_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>
#include <ardrone_autonomy/vector31.h>
#include <ardrone_autonomy/vector21.h>

namespace ardrone_autonomy
{
template <class ContainerAllocator>
struct navdata_altitude_
{
  typedef navdata_altitude_<ContainerAllocator> Type;

  navdata_altitude_()
    : header()
    , drone_time(0.0)
    , tag(0)
    , size(0)
    , altitude_vision(0)
    , altitude_vz(0.0)
    , altitude_ref(0)
    , altitude_raw(0)
    , obs_accZ(0.0)
    , obs_alt(0.0)
    , obs_x()
    , obs_state(0)
    , est_vb()
    , est_state(0)  {
    }
  navdata_altitude_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , drone_time(0.0)
    , tag(0)
    , size(0)
    , altitude_vision(0)
    , altitude_vz(0.0)
    , altitude_ref(0)
    , altitude_raw(0)
    , obs_accZ(0.0)
    , obs_alt(0.0)
    , obs_x(_alloc)
    , obs_state(0)
    , est_vb(_alloc)
    , est_state(0)  {
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

   typedef int32_t _altitude_vision_type;
  _altitude_vision_type altitude_vision;

   typedef float _altitude_vz_type;
  _altitude_vz_type altitude_vz;

   typedef int32_t _altitude_ref_type;
  _altitude_ref_type altitude_ref;

   typedef int32_t _altitude_raw_type;
  _altitude_raw_type altitude_raw;

   typedef float _obs_accZ_type;
  _obs_accZ_type obs_accZ;

   typedef float _obs_alt_type;
  _obs_alt_type obs_alt;

   typedef  ::ardrone_autonomy::vector31_<ContainerAllocator>  _obs_x_type;
  _obs_x_type obs_x;

   typedef uint32_t _obs_state_type;
  _obs_state_type obs_state;

   typedef  ::ardrone_autonomy::vector21_<ContainerAllocator>  _est_vb_type;
  _est_vb_type est_vb;

   typedef uint32_t _est_state_type;
  _est_state_type est_state;




  typedef boost::shared_ptr< ::ardrone_autonomy::navdata_altitude_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::ardrone_autonomy::navdata_altitude_<ContainerAllocator> const> ConstPtr;

}; // struct navdata_altitude_

typedef ::ardrone_autonomy::navdata_altitude_<std::allocator<void> > navdata_altitude;

typedef boost::shared_ptr< ::ardrone_autonomy::navdata_altitude > navdata_altitudePtr;
typedef boost::shared_ptr< ::ardrone_autonomy::navdata_altitude const> navdata_altitudeConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::ardrone_autonomy::navdata_altitude_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::ardrone_autonomy::navdata_altitude_<ContainerAllocator> >::stream(s, "", v);
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
struct IsFixedSize< ::ardrone_autonomy::navdata_altitude_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::ardrone_autonomy::navdata_altitude_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::ardrone_autonomy::navdata_altitude_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::ardrone_autonomy::navdata_altitude_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::ardrone_autonomy::navdata_altitude_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::ardrone_autonomy::navdata_altitude_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::ardrone_autonomy::navdata_altitude_<ContainerAllocator> >
{
  static const char* value()
  {
    return "f0fd1fd20697e6083c6bc3a308a260dc";
  }

  static const char* value(const ::ardrone_autonomy::navdata_altitude_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xf0fd1fd20697e608ULL;
  static const uint64_t static_value2 = 0x3c6bc3a308a260dcULL;
};

template<class ContainerAllocator>
struct DataType< ::ardrone_autonomy::navdata_altitude_<ContainerAllocator> >
{
  static const char* value()
  {
    return "ardrone_autonomy/navdata_altitude";
  }

  static const char* value(const ::ardrone_autonomy::navdata_altitude_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::ardrone_autonomy::navdata_altitude_<ContainerAllocator> >
{
  static const char* value()
  {
    return "Header header\n\
float64 drone_time\n\
uint16 tag\n\
uint16 size\n\
int32 altitude_vision\n\
float32 altitude_vz\n\
int32 altitude_ref\n\
int32 altitude_raw\n\
float32 obs_accZ\n\
float32 obs_alt\n\
vector31 obs_x\n\
uint32 obs_state\n\
vector21 est_vb\n\
uint32 est_state\n\
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
================================================================================\n\
MSG: ardrone_autonomy/vector21\n\
float32 x\n\
float32 y\n\
";
  }

  static const char* value(const ::ardrone_autonomy::navdata_altitude_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::ardrone_autonomy::navdata_altitude_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.drone_time);
      stream.next(m.tag);
      stream.next(m.size);
      stream.next(m.altitude_vision);
      stream.next(m.altitude_vz);
      stream.next(m.altitude_ref);
      stream.next(m.altitude_raw);
      stream.next(m.obs_accZ);
      stream.next(m.obs_alt);
      stream.next(m.obs_x);
      stream.next(m.obs_state);
      stream.next(m.est_vb);
      stream.next(m.est_state);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct navdata_altitude_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::ardrone_autonomy::navdata_altitude_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::ardrone_autonomy::navdata_altitude_<ContainerAllocator>& v)
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
    s << indent << "altitude_vision: ";
    Printer<int32_t>::stream(s, indent + "  ", v.altitude_vision);
    s << indent << "altitude_vz: ";
    Printer<float>::stream(s, indent + "  ", v.altitude_vz);
    s << indent << "altitude_ref: ";
    Printer<int32_t>::stream(s, indent + "  ", v.altitude_ref);
    s << indent << "altitude_raw: ";
    Printer<int32_t>::stream(s, indent + "  ", v.altitude_raw);
    s << indent << "obs_accZ: ";
    Printer<float>::stream(s, indent + "  ", v.obs_accZ);
    s << indent << "obs_alt: ";
    Printer<float>::stream(s, indent + "  ", v.obs_alt);
    s << indent << "obs_x: ";
    s << std::endl;
    Printer< ::ardrone_autonomy::vector31_<ContainerAllocator> >::stream(s, indent + "  ", v.obs_x);
    s << indent << "obs_state: ";
    Printer<uint32_t>::stream(s, indent + "  ", v.obs_state);
    s << indent << "est_vb: ";
    s << std::endl;
    Printer< ::ardrone_autonomy::vector21_<ContainerAllocator> >::stream(s, indent + "  ", v.est_vb);
    s << indent << "est_state: ";
    Printer<uint32_t>::stream(s, indent + "  ", v.est_state);
  }
};

} // namespace message_operations
} // namespace ros

#endif // ARDRONE_AUTONOMY_MESSAGE_NAVDATA_ALTITUDE_H
