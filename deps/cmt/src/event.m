/*
 * Copyright (c), Recep Aslantas.
 * MIT License (MIT), http://opensource.org/licenses/MIT
 */

#include "impl/common.h"
#include "cmt/common.h"

CF_RETURNS_RETAINED
MT_EXPORT
MT_API_AVAILABLE(mt_macos(10.14), mt_ios(12.0))
MtEvent*
mtDeviceNewEvent(MtDevice *dev) {
	return [(id<MTLDevice>)dev newEvent];
}

MT_EXPORT
MT_API_AVAILABLE(mt_macos(10.14), mt_ios(12.0))
MtSharedEvent*
mtDeviceNewSharedEvent(MtDevice *dev) {
	return [(id<MTLDevice>)dev newSharedEvent];
}

MT_EXPORT
MT_API_AVAILABLE(mt_macos(10.14), mt_ios(12.0))
MtSharedEvent*
mtDeviceNewSharedEventWithHandle(MtDevice *dev, MtSharedEventHandle *handle) {
	return [(id<MTLDevice>)dev newSharedEventWithHandle: (MTLSharedEventHandle*)handle];
}

MT_EXPORT
MT_API_AVAILABLE(mt_macos(10.13), mt_ios(10.0))
MtFence*
mtDeviceNewFence(MtDevice *dev) {
	return [(id<MTLDevice>)dev newFence];
}

MT_EXPORT
MT_API_AVAILABLE(mt_macos(10.14), mt_ios(12.0))
MtDevice*
mtEventDevice(MtEvent *event) {
	return [(id<MTLEvent>)event device];
}

MT_EXPORT
MT_API_AVAILABLE(mt_macos(10.14), mt_ios(12.0))
const char*
mtEventLabel(MtEvent *event) {
	return Cstring([(id<MTLEvent>)event label]);
}

MT_EXPORT
MT_API_AVAILABLE(mt_macos(10.14), mt_ios(12.0))
void
mtEventLabelSet(MtEvent *event, const char* label) {
	((id<MTLEvent>)event).label = mtNSString(label);
}

// shared
MT_EXPORT
MT_API_AVAILABLE(mt_macos(10.14), mt_ios(12.0))
uint64_t
mtSharedEventSignaledValue(MtSharedEvent *event) {
	return [(id<MTLSharedEvent>)event signaledValue];
}

// shared
MT_EXPORT
MT_API_AVAILABLE(mt_macos(10.14), mt_ios(12.0))
MtSharedEventHandle*
mtSharedEventNewHandle(MtSharedEvent *event) {
	return [(id<MTLSharedEvent>)event newSharedEventHandle];
}

MT_EXPORT
MT_API_AVAILABLE(mt_macos(10.14), mt_ios(12.0))
void
mtSharedEventNotifyListener(MtSharedEvent *event, MtSharedEventListener *listener, uint64_t val, MtSharedEventNotificationBlock block) {
	[(id<MTLSharedEvent>)event notifyListener: (MTLSharedEventListener*)listener
										atValue:val
										block: (MTLSharedEventNotificationBlock) block];
}

MT_EXPORT
MT_API_AVAILABLE(mt_macos(10.14), mt_ios(12.0))
void
mtSharedEventWait(MtSharedEvent *event, uint64_t val) {
  dispatch_semaphore_t sema = dispatch_semaphore_create(0);
  dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0);
  MTLSharedEventListener * sharedEventListener = [[MTLSharedEventListener alloc] initWithDispatchQueue:queue];

  [(id<MTLSharedEvent>)event notifyListener:sharedEventListener
                       				atValue:val
                         			  block:^(id<MTLSharedEvent> event, uint64_t value) {
    dispatch_semaphore_signal(sema);
  }];
    
  dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);
}

MT_EXPORT
MT_API_AVAILABLE(mt_macos(10.14), mt_ios(12.0))
void
mtSharedEventSignal(MtSharedEvent *event, uint64_t val) {
  ((id<MTLSharedEvent>) event).signaledValue = val;
}