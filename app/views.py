from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
from app.forms import *
from app.models import Leave, LeaveApprovingWarden, LeaveApprovingFaculty
from datetime import datetime
from app.utils.cosineSimilarity import *
from app.utils.tfIDF import *

def index(request):
    if request.user.is_authenticated:
        return redirect('app:dashboard')
    return redirect('users:login')


def profile(request):
    user = request.user
    try:
        user_account = request.user.user_account

        if user_account.user_type == 'S':
            student = user_account.student
            program_type = {
                'BTC': 'Bachelor of Technology',
                'MTC': 'Master of Technology',
                'MSC': 'Master of Science',
                'PHD': 'Doctorate of Philosophy',
                'OTH': 'Other',
            }
            discipline_type = {
                'CSE': 'Computer Science and Engineering',
                'EE': 'Electrical Engineering',
                'CEE': 'Civil and Environmental Engineering',
                'CBE': 'Chemical and Biochemical Engineering',
                'ME': 'Mechanical Engineering',
                'MME': 'Metallurgical and Materials Engineering',
                'MSE': 'Material Science and Engineering',
                'CHEM': 'Chemistry',
                'MATHS': 'Mathematics',
                'PHY': 'Physics',
                'HSS': 'Humanities and Social Sciences',
                'MNC': 'Mathematics and Computing',
                'MT': 'Mechatronics',
                'NT': 'Nanoscience and Technology',
                'CM': 'Communication System engineering',
                'OTH': 'Other',
            }
            gender_type = {
                'M': 'Male',
                'F': 'Female',
                'O': 'Other',
            }
            hostel_type = {
                'GH': 'Girls Hostel',
                'BH': 'Boys Hostel',
            }
            context = {'student': student, 'program_type': program_type, 'discipline_type': discipline_type,
                       'gender_type': gender_type, 'hostel_type': hostel_type}

            if request.method == 'POST':
                user.first_name = request.POST.get('first_name', '')
                user.last_name = request.POST.get('last_name', '')
                user.save()

                student.current_year = request.POST.get('current_year', '')
                student.block = request.POST.get('block', '')
                student.room_number = request.POST.get('room_number', '')
                student.save()
                context['message'] = 'Your profile has been successfully saved!'
            return render(request, 'app/student_profile.html', context)

        elif user_account.user_type == 'AA':
            designation_type = {
                'ASIP': 'Assistant Professor',
                'ASOP': 'Associate Professor',
                'PROF': 'Professor',
                'OTH': 'Other',
            }
            role_type = {
                'FAD': 'Faculty Advisor',
                'GHW': 'Warden - Girls Hostel',
                'BHW': 'Warden - Boys Hostel',
                'SUP': 'Supervisor',
                'OTH': 'Other',
            }
            authority = user_account.authority
            context = {'authority': authority, 'designation_type': designation_type, 'role_type': role_type}

            if request.method == 'POST':
                user.first_name = request.POST.get('first_name', '')
                user.last_name = request.POST.get('last_name', '')
                user.save()
                context['message'] = 'Your profile has been successfully saved!'

            return render(request, 'app/approving_authority_profile.html', context)
    except:
        return redirect(request, 'app/error.html', {})


def change_password(request):
    message = ''
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)
            message = 'Your password was successfully updated!'
            return render(request, 'app/change_password.html', {
                'form': form,
                'message': message
            })
        else:
            message = 'Please enter valid credentials'
    else:
        form = PasswordChangeForm(request.user)
    return render(request, 'app/change_password.html', {
        'form': form,
        'message': message
    })

def docSimilarity(request):
    
    if request.method == 'POST':
        # fm = DocSimilarity()
        data = request.POST
        data1 = data['data1'].lower()
        data2 = data['data2'].lower()
        algorithm = data['algorithm']
        score = -1;
        if algorithm == 'cosineSimilarity':
            score = cosineSimilarity(data1, data2)
        elif algorithm == 'tfIDF':
            score = tfIDF(data1, data2)
        else:
            pass

            # score = 
        print(score)
        # return redirect(request, 'app/error.html', {})
        # if score:
        return render(request, 'app/showResult.html', {'score' : score, 'algorithm' : algorithm})
    else:
        fm = DocSimilarity()
        return render(request, 'app/docSimilarity.html', {'form' : fm})

def dashboard(request):
    user_account = request.user.user_account
    if user_account.user_type == 'S':
        return render(request, 'app/student_dashboard.html', {})
    else:
        return render(request, 'app/authority_dashboard.html', {})


def leave_create(request):
    student = request.user.user_account.student
    if request.method == "POST":
        format_str = '%m/%d/%Y'
        dol_str = request.POST.get('date_of_leaving', '')
        dor_str = request.POST.get('date_of_returning', '')
        dol_obj = datetime.strptime(dol_str, format_str)
        dor_obj = datetime.strptime(dor_str, format_str)
        leave = Leave.objects.create(
            reason_for_leave=request.POST.get('reason_for_leave', ''),
            going_to_place=request.POST.get('going_to_place', ''),
            going_to_type=request.POST.get('going_type', ''),
            date_of_leaving=dol_obj.date(),
            date_of_returning=dor_obj.date()
        )
        leave.student = request.user.user_account.student
        leave.faculty = \
            LeaveApprovingFaculty.objects.filter(batch=student.year_of_joining).filter(program=student.program).filter(
                discipline=student.discipline)[0]
        leave.warden = LeaveApprovingWarden.objects.filter(hostel=student.hostel)[0]
        leave.save()
        return redirect('app:leave_detail', pk=leave.pk)
    else:
        form = LeaveForm()
    return render(request, 'app/leave_create.html', {'form': form})


def leave_detail(request, pk):
    leave = get_object_or_404(Leave, pk=pk)
    user_account = request.user.user_account
    if request.method == "POST":
        if user_account.user_type == 'AA':
            if user_account.authority.role == 'FAD':
                leave.leave_status = request.POST.get('leave_status', '')
            else:
                leave.leave_status = request.POST.get('leave_status', '')
        else:
            leave.leave_status = 'PEN'
        leave.save()
        return redirect('app:dashboard')
    else:
        leave_status_values = {
            'PEN': 'Pending',
            'APPF': 'Approved by Faculty',
            'APPW': 'Approved by Warden',
            'REJ': "Rejected",
        }
        hostel_type = {
            'GH': 'Girls Hostel',
            'BH': 'Boys Hostel',
        }
        if user_account.user_type == 'S':
            return render(request, 'app/leave_detail.html', {'leave': leave, 'can_edit': True, 'can_approve': False,
                                                             'leave_status_values': leave_status_values, 'hostel_type': hostel_type})
        else:
            if user_account.authority.role == 'FAD':
                return render(request, 'app/leave_detail.html',
                              {'leave': leave, 'can_edit': False, 'can_approve': True, 'is_warden': False,
                               'leave_status_values': leave_status_values, 'hostel_type': hostel_type})
            else:
                return render(request, 'app/leave_detail.html',
                              {'leave': leave, 'can_edit': False, 'can_approve': True, 'is_warden': True,
                               'leave_status_values': leave_status_values, 'hostel_type': hostel_type})


def leave_edit(request, pk):
    leave = get_object_or_404(Leave, pk=pk)
    if request.method == "POST":
        leave.reason_for_leave = request.POST.get('reason_for_leave', '')
        leave.going_to_place = request.POST.get('going_to_place', '')
        leave.going_to_type = request.POST.get('going_to_type', '')
        leave.date_of_leaving = request.POST.get('date_of_leaving', '')
        leave.date_of_returning = request.POST.get('date_of_returning', '')
        if request.user.user_account.user_type == 'S':
            leave.student = request.user.user_account.student
        else:
            leave.leave_status = request.POST.get('leave_status', 'PEN')
        leave.save()
        return redirect('app:leave_detail', pk=leave.pk)
    else:
        return render(request, 'app/leave_edit.html', {'leave': leave })


def leaves_pending(request):
    user_account = request.user.user_account
    leave_status_values = {
        'PEN': 'Pending',
        'APPF': 'Approved by Faculty',
        'APPW': 'Approved by Warden',
        'REJ': "Rejected",
    }
    if user_account.user_type == 'S':
        leaves = Leave.objects.filter(student=user_account.student).exclude(leave_status="APPW").exclude(leave_status="REJ").order_by("-pk")
        return render(request, 'app/leaves_list.html', {'leaves': leaves, 'can_edit': True, 'can_approve': False, 'leave_status_values': leave_status_values})
    else:
        authority_type = user_account.authority.role
        try:
            if authority_type == 'FAD':
                leaves = Leave.objects.filter(faculty=user_account.authority.faculty).filter(leave_status="PEN").order_by("-pk")
            else:
                leaves = Leave.objects.filter(warden=user_account.authority.warden).filter(leave_status="APPF").order_by("-pk")
        except:
            leaves = []
        return render(request, 'app/leaves_list.html', {'leaves': leaves, 'can_edit': True, 'can_approve': True, 'leave_status_values': leave_status_values})


def leaves_past(request):
    user_account = request.user.user_account
    leave_status_values = {
        'PEN': 'Pending',
        'APPF': 'Approved by Faculty',
        'APPW': 'Approved by Warden',
        'REJ': "Rejected",
    }
    if user_account.user_type == 'S':
        leaves = Leave.objects.filter(student=user_account.student).exclude(leave_status="PEN").exclude(leave_status="APPF").order_by("-pk")
        return render(request, 'app/leaves_list.html', {'leaves': leaves, 'can_edit': False, 'leave_status_values': leave_status_values,})
    else:
        authority_type = user_account.authority.role
        try:
            if authority_type == 'FAD':
                leaves = Leave.objects.filter(faculty=user_account.authority.faculty).exclude(leave_status="PEN").order_by("-pk")
            else:
                leaves = Leave.objects.filter(warden=user_account.authority.warden).exclude(
                    leave_status="PEN").exclude(leave_status="APPF").order_by("-pk")
        except:
            leaves = []
        return render(request, 'app/leaves_list.html', {'leaves': leaves, 'can_edit': False, 'leave_status_values': leave_status_values})
