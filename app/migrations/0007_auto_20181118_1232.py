# Generated by Django 2.1.3 on 2018-11-18 12:32

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0006_auto_20181118_1229'),
    ]

    operations = [
        migrations.AlterField(
            model_name='leave',
            name='warden',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='app.LeaveApprovingWarden'),
        ),
    ]
