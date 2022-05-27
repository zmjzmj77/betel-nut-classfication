
#pragma once

#include "Infra/Guard.h"
#include "Infra/Semaphore.h"
#include <list>
#include"GenICam/System.h"

using namespace Dahua::GenICam;
using namespace Dahua::Infra;

template<typename T>
class TMessageQue
{
public:

    void push_back(T const& msg)
    {
        Dahua::Infra::CGuard g(_mutex);
        _list.push_back(msg);
        _sem.post();
    }

    void get(T& msg)
    {
        _mutex.enter();     //该线程获取互斥体
        while (_list.empty())
        {
            _mutex.leave();
            _sem.pend();
            _mutex.enter();
        }

        msg = _list.front();//返回链表中的第一个元素值
        _list.pop_front();//删除1个元素
        _mutex.leave();//释放互斥体
    }

    bool get(T& msg, int timeout)
    {
        _mutex.enter();
        while (_list.empty())
        {
            _mutex.leave();
            if (_sem.pend(timeout) < 0)
            {
                return false;
            }
            _mutex.enter();
        }

        msg = _list.front();
//        free(_list.front().m_pImageBuf);
        _list.pop_front();
        _mutex.leave();
        return true;
    }

    int size()
    {
        Dahua::Infra::CGuard g(_mutex);
        return _list.size();
    }

    void clear()
    {
        Dahua::Infra::CGuard g(_mutex);
        _list.clear();
    }
private:

    Dahua::Infra::CSemaphore _sem;
    Dahua::Infra::CMutex     _mutex;
    std::list<T>             _list;
};
