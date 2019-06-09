from django.conf.urls import url
from . import views

from django.conf import settings # add
from django.conf.urls.static import static # add

urlpatterns = [
  url(r'^uimage/$', views.uimage, name='uimage'), # add
  url(r'^$', views.first_page, name='first_page'), # add
  url(r'^pixelize/$', views.pixelization,name='pixelization'),
  url(r'^dface/$', views.dface,name='dface')
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) # add
